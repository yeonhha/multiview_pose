import torch
import torch.nn as nn
import torch.nn.functional as F
from multiview_pose.core.camera import CustomSimpleCameraTorch as SimpleCameraTorch
from multiview_pose.core.post_processing.post_transforms import transform_preds_torch
from multiview_pose.core.camera.utils import StereoGeometry, MonocularGeometry
from itertools import combinations, product
from .utils import compute_grid
from .builder import GCNS
from matplotlib import pyplot as plt
import numpy as np
import mmcv
import os

def imshow_multiview_joints_3d(
    pose_result,
    space_size=[8000, 8000, 2000],
    space_center=[0, -500, 800]
):
    """Draw 3D keypoints and links in 3D coordinates.

    Args:
        pose_result (list[kpts]): The poses to draw. Each element kpts is
            a set of K keypoints as an Kx4 numpy.ndarray, where each
            keypoint is represented as x, y, z, score.
        skeleton (list of [idx_i,idx_j]): Skeleton described by a list of
            links, each is a pair of joint indices.
        pose_kpt_color (np.ndarray[Nx3]`): Color of N keypoints. If None, do
            not nddraw keypoints.
        pose_link_color (np.array[Mx3]): Color of M links. If None, do not
            draw links.
        space_size: (list). Default: [8000, 8000, 2000].
        space_center: (list). Default: [0, -500, 800].
        kpt_score_thr (float): Minimum score of keypoints to be shown.
            Default: 0.0.
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(space_center[0] - space_size[0] * 0.5,
                  space_center[0] + space_size[0] * 0.5)
    ax.set_ylim3d(space_center[1] - space_size[1] * 0.5,
                  space_center[1] + space_size[1] * 0.5)
    ax.set_zlim3d(space_center[2] - space_size[2] * 0.5,
                  space_center[2] + space_size[2] * 0.5)


    for kpts in pose_result:
        # draw each point on image
        xs, ys, zs = kpts.T
        # valid = scores > kpt_score_thr
        ax.scatter(
            xs,
            ys,
            zs,
            marker='o',
            color='limegreen')

    # convert figure to numpy array
    fig.tight_layout()
    fig.canvas.draw()
    img_w, img_h = fig.canvas.get_width_height()
    img_vis = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(img_h, img_w, -1)
    img_vis = mmcv.rgb2bgr(img_vis)

    plt.close(fig)

    return img_vis

    
@GCNS.register_module()
class MultiViewMatchModule(nn.Module):
    def __init__(self, feature_map_size, match_gcn, num_cameras=5,
                 match_threshold=0.5,
                 cfg_2d=dict(center_index=2,
                             nms_kernel=5,
                             val_threshold=0.3,
                             dist_threshold=5,
                             max_persons=10,
                             center_channel=512 + 2,
                             dist_coef=10),
                 cfg_3d=dict(space_size=[8000, 8000, 2000],
                             space_center=[0, -500, 800],
                             cube_size=[80, 80, 20],
                             dist_threshold=300)):
        self.cfg_2d = cfg_2d.copy()
        self.cfg_3d = cfg_3d.copy()
        self.match_threshold = match_threshold
        super(MultiViewMatchModule, self).__init__()
        self.register_buffer('feature_map_size', torch.tensor(feature_map_size))
        self.register_buffer('grid_samples', compute_grid(cfg_3d['space_size'],
                                                          cfg_3d['space_center'],
                                                          cfg_3d['cube_size']))
        self.loss = nn.BCELoss(reduction='none')
        self.match_gcn = GCNS.build(match_gcn)
        self.num_cameras = num_cameras

        # self._initial_edge_indices()
        self.pool = torch.nn.MaxPool2d(cfg_2d['nms_kernel'], 1,
                                       (cfg_2d['nms_kernel'] - 1) // 2)
        self._initialize_connections()

    def _get_sparse_candidates(self, samples_to_query):
        L, W, H = self.cfg_3d['cube_size']
        samples_to_query = samples_to_query.view(L, W, H)
        samples_to_query[::2, ::2, ::2] = 1
        return samples_to_query.view(-1)

    def _initialize_connections(self):
        camera_pairs = torch.tensor(
            list(combinations(range(self.num_cameras), 2)))
        person_pairs = torch.tensor(
            list(product(range(self.cfg_2d['max_persons']), repeat=2)))

        self.register_buffer('camera_pairs', torch.tensor(camera_pairs))
        self.register_buffer('person_pairs', torch.tensor(person_pairs))

    def forward(self, feature_maps, img_metas, return_loss=True, graph=None):
        """

        Args:
            feature_maps: [NCHW]xV -> NxVxCxHxW
            img_metas:
        Returns:
            center_candidates list: [num_candidates_i x 5] i=0:N-1
        """
        feature_maps = torch.stack(feature_maps, dim=1)  # NxVxCxHxW
        if return_loss:
            # debug start
            feature_maps = feature_maps.detach()
            # debug end
            graph = self.build_graph_from_input(feature_maps, graph)
            return self.forward_train(graph)
        else:
            return self.forward_test(feature_maps, img_metas)

    def nms(self, heatmaps):
        """Non-Maximum Suppression for heatmaps.

        Args:
            heatmap(torch.Tensor): Heatmaps before nms.

        Returns:
            torch.Tensor: Heatmaps after nms.
        """

        maxm = self.pool(heatmaps)
        maxm = torch.eq(maxm, heatmaps).float()
        heatmaps = heatmaps * maxm

        return heatmaps

    def top_k(self, heatmaps):
        """Find top_k values in an image.
        Args:
            heatmaps (torch.Tensor[NxVxHxW])

        Return:
        """
        heatmaps = self.nms(heatmaps)
        N, V, H, W = heatmaps.shape
        heatmaps = heatmaps.view(N, V, -1)
        val_k, ind = heatmaps.topk(self.cfg_2d['max_persons'],
                                   dim=2)  # NxVxP
        x = ind % W
        y = ind // W
        ind_k = torch.stack((x, y), dim=3)  # NxVxPx2

        return ind_k, val_k

    def forward_train(self, graph):
        # edge_indices = graph['edge_indices'][graph['edge_valid'][:, 0] > 0].long()

        neck_edge_indices = graph['edge_indices'][graph['neck_edge_valid'][:, 0] > 0].long()
        nose_edge_indices = graph['edge_indices'][graph['nose_edge_valid'][:, 0] > 0].long()
        hip_edge_indices = graph['edge_indices'][graph['hip_edge_valid'][:, 0] > 0].long()
        lshoulder_edge_indices = graph['edge_indices'][graph['lshoulder_edge_valid'][:, 0] > 0].long()
        lelbow_edge_indices = graph['edge_indices'][graph['lelbow_edge_valid'][:, 0] > 0].long()
        lwrist_edge_indices = graph['edge_indices'][graph['lwrist_edge_valid'][:, 0] > 0].long()
        lhip_edge_indices = graph['edge_indices'][graph['lhip_edge_valid'][:, 0] > 0].long()
        lknee_edge_indices = graph['edge_indices'][graph['lknee_edge_valid'][:, 0] > 0].long()
        lankle_edge_indices = graph['edge_indices'][graph['lankle_edge_valid'][:, 0] > 0].long()
        rshoulder_edge_indices = graph['edge_indices'][graph['rshoulder_edge_valid'][:, 0] > 0].long()
        relbow_edge_indices = graph['edge_indices'][graph['relbow_edge_valid'][:, 0] > 0].long()
        rwrist_edge_indices = graph['edge_indices'][graph['rwrist_edge_valid'][:, 0] > 0].long()
        rhip_edge_indices = graph['edge_indices'][graph['rhip_edge_valid'][:, 0] > 0].long()
        rknee_edge_indices = graph['edge_indices'][graph['rknee_edge_valid'][:, 0] > 0].long()
        rankle_edge_indices = graph['edge_indices'][graph['rankle_edge_valid'][:, 0] > 0].long()

        # edge_scores = graph['edge_scores'][graph['edge_valid'][:, 0] > 0]

        neck_edge_scores = graph['neck_edge_scores'][graph['neck_edge_valid'][:, 0] > 0]
        nose_edge_scores = graph['nose_edge_scores'][graph['nose_edge_valid'][:, 0] > 0]
        hip_edge_scores = graph['hip_edge_scores'][graph['hip_edge_valid'][:, 0] > 0]
        lshoulder_edge_scores = graph['lshoulder_edge_scores'][graph['lshoulder_edge_valid'][:, 0] > 0]
        lelbow_edge_scores = graph['lelbow_edge_scores'][graph['lelbow_edge_valid'][:, 0] > 0]
        lwrist_edge_scores = graph['lwrist_edge_scores'][graph['lwrist_edge_valid'][:, 0] > 0]
        lhip_edge_scores = graph['lhip_edge_scores'][graph['lhip_edge_valid'][:, 0] > 0]
        lknee_edge_scores = graph['lknee_edge_scores'][graph['lknee_edge_valid'][:, 0] > 0]
        lankle_edge_scores = graph['lankle_edge_scores'][graph['lankle_edge_valid'][:, 0] > 0]
        rshoulder_edge_scores = graph['rshoulder_edge_scores'][graph['rshoulder_edge_valid'][:, 0] > 0]
        relbow_edge_scores = graph['relbow_edge_scores'][graph['relbow_edge_valid'][:, 0] > 0]
        rwrist_edge_scores = graph['rwrist_edge_scores'][graph['rwrist_edge_valid'][:, 0] > 0]
        rhip_edge_scores = graph['rhip_edge_scores'][graph['rhip_edge_valid'][:, 0] > 0]
        rknee_edge_scores = graph['rknee_edge_scores'][graph['rknee_edge_valid'][:, 0] > 0]
        rankle_edge_scores = graph['rankle_edge_scores'][graph['rankle_edge_valid'][:, 0] > 0]

        # edge_labels = graph['edge_labels'][graph['edge_valid'][:, 0] > 0]
        neck_edge_labels = graph['edge_labels'][graph['neck_edge_valid'][:, 0] > 0]
        nose_edge_labels = graph['edge_labels'][graph['nose_edge_valid'][:, 0] > 0]
        hip_edge_labels = graph['edge_labels'][graph['hip_edge_valid'][:, 0] > 0]
        lshoulder_edge_labels = graph['edge_labels'][graph['lshoulder_edge_valid'][:, 0] > 0]
        lelbow_edge_labels = graph['edge_labels'][graph['lelbow_edge_valid'][:, 0] > 0]
        lwrist_edge_labels = graph['edge_labels'][graph['lwrist_edge_valid'][:, 0] > 0]
        lhip_edge_labels = graph['edge_labels'][graph['lhip_edge_valid'][:, 0] > 0]
        lknee_edge_labels = graph['edge_labels'][graph['lknee_edge_valid'][:, 0] > 0]
        lankle_edge_labels = graph['edge_labels'][graph['lankle_edge_valid'][:, 0] > 0]
        rshoulder_edge_labels = graph['edge_labels'][graph['rshoulder_edge_valid'][:, 0] > 0]
        relbow_edge_labels = graph['edge_labels'][graph['relbow_edge_valid'][:, 0] > 0]
        rwrist_edge_labels = graph['edge_labels'][graph['rwrist_edge_valid'][:, 0] > 0]
        rhip_edge_labels = graph['edge_labels'][graph['rhip_edge_valid'][:, 0] > 0]
        rknee_edge_labels = graph['edge_labels'][graph['rknee_edge_valid'][:, 0] > 0]
        rankle_edge_labels = graph['edge_labels'][graph['rankle_edge_valid'][:, 0] > 0]


        if (graph['neck_edge_valid'][:, 0] > 0).sum() >= 1:
            # _, preds = self.match_gcn(graph['node_features'], edge_indices, edge_scores)
            # preds = preds.sigmoid()

            _, neck_preds = self.match_gcn(graph['neck_node_features'], neck_edge_indices, neck_edge_scores)
            neck_preds = neck_preds.sigmoid()

            _, nose_preds = self.match_gcn(graph['nose_node_features'], nose_edge_indices,nose_edge_scores)
            nose_preds = nose_preds.sigmoid()

            _, hip_preds = self.match_gcn(graph['hip_node_features'], hip_edge_indices, hip_edge_scores)
            hip_preds = hip_preds.sigmoid()

            _, lshoulder_preds = self.match_gcn(graph['lshoulder_node_features'], lshoulder_edge_indices, lshoulder_edge_scores)
            lshoulder_preds = lshoulder_preds.sigmoid()

            _, lelbow_preds = self.match_gcn(graph['lelbow_node_features'], lelbow_edge_indices, lelbow_edge_scores)
            lelbow_preds = lelbow_preds.sigmoid()

            _, lwrist_preds = self.match_gcn(graph['lwrist_node_features'], lwrist_edge_indices, lwrist_edge_scores)
            lwrist_preds = lwrist_preds.sigmoid() 

            _, lhip_preds = self.match_gcn(graph['lhip_node_features'], lhip_edge_indices, lhip_edge_scores)
            lhip_preds = lhip_preds.sigmoid()

            _, lknee_preds = self.match_gcn(graph['lknee_node_features'], lknee_edge_indices, lknee_edge_scores)
            lknee_preds = lknee_preds.sigmoid()

            _, lankle_preds = self.match_gcn(graph['lankle_node_features'], lankle_edge_indices, lankle_edge_scores)
            lankle_preds = lankle_preds.sigmoid()

            _, rshoulder_preds = self.match_gcn(graph['rshoulder_node_features'], rshoulder_edge_indices, rshoulder_edge_scores)
            rshoulder_preds = rshoulder_preds.sigmoid()

            _, relbow_preds = self.match_gcn(graph['relbow_node_features'], relbow_edge_indices, relbow_edge_scores)
            relbow_preds = relbow_preds.sigmoid()

            _, rwrist_preds = self.match_gcn(graph['rwrist_node_features'], rwrist_edge_indices, rwrist_edge_scores)
            rwrist_preds = rwrist_preds.sigmoid()

            _, rhip_preds = self.match_gcn(graph['rhip_node_features'], rhip_edge_indices, rhip_edge_scores)
            rhip_preds = rhip_preds.sigmoid()

            _, rknee_preds = self.match_gcn(graph['rknee_node_features'], rknee_edge_indices, rknee_edge_scores)
            rknee_preds = rknee_preds.sigmoid()

            _, rankle_preds = self.match_gcn(graph['rankle_node_features'], rankle_edge_indices, rankle_edge_scores)
            rankle_preds = rankle_preds.sigmoid()

            loss = self.get_loss(neck_preds, neck_edge_labels) + self.get_loss(nose_preds, nose_edge_labels) + self.get_loss(hip_preds, hip_edge_labels) + self.get_loss(lshoulder_preds, lshoulder_edge_labels) \
            + self.get_loss(lelbow_preds, lelbow_edge_labels) + self.get_loss(lwrist_preds, lwrist_edge_labels) + self.get_loss(lhip_preds, lhip_edge_labels) + self.get_loss(lknee_preds, lknee_edge_labels) \
            + self.get_loss(lankle_preds, lankle_edge_labels) + self.get_loss(rshoulder_preds, rshoulder_edge_labels) + self.get_loss(relbow_preds, relbow_edge_labels) + self.get_loss(rwrist_preds, rwrist_edge_labels) \
            + self.get_loss(rhip_preds, rhip_edge_labels) + self.get_loss(rknee_preds, rknee_edge_labels) + self.get_loss(rankle_preds, rankle_edge_labels)
            
            
            # loss_match=self.get_loss(neck_preds, neck_edge_labels)
            loss_match = loss / 15

            return dict(loss_match= loss / 15)

        else:
            return dict(loss_match=graph['edge_scores'].new_zeros(1)[0])

    def get_loss(self, preds, edge_labels):
        num_positives = edge_labels.sum()
        num_samples = edge_labels.shape[0]
        num_negatives = num_samples - num_positives
        loss = self.loss(preds, edge_labels)
        if (edge_labels > 0).sum() < 1:
            print(edge_labels.shape, flush=True)
        ones = torch.ones_like(edge_labels)
        mask = torch.where(edge_labels > 0, ones / (num_positives + 1e-12), ones / (num_negatives + 1e-12))

        return (loss * mask).sum()

    def build_graph_from_input(self, feature_maps, graph):
        """

        Args:
            feature_maps: NxVxCxHxW
            graph:
                multiview_centers: Nx(VxP)x3
                edge_indices: NxEx2
                edge_scores: NxE
                edge_valid: NxE
                edge_labels: NxE

        Returns:
        """
        batch_size, num_nodes, _ = graph['multiview_necks'].shape
        for i in range(batch_size):
            graph['edge_indices'][i] = graph['edge_indices'][i] + num_nodes * i

        graph['edge_indices'] = graph['edge_indices'].view(-1, 2)
        
        
        graph['neck_edge_scores'] = graph['neck_edge_scores'].view(-1, 1)
        graph['nose_edge_scores'] = graph['nose_edge_scores'].view(-1, 1)
        graph['hip_edge_scores'] = graph['hip_edge_scores'].view(-1, 1)
        graph['lshoulder_edge_scores'] = graph['lshoulder_edge_scores'].view(-1, 1)
        graph['lelbow_edge_scores'] = graph['lelbow_edge_scores'].view(-1, 1)
        graph['lwrist_edge_scores'] = graph['lwrist_edge_scores'].view(-1, 1)
        graph['lhip_edge_scores'] = graph['lhip_edge_scores'].view(-1, 1)
        graph['lknee_edge_scores'] = graph['lknee_edge_scores'].view(-1, 1)
        graph['lankle_edge_scores'] = graph['lankle_edge_scores'].view(-1, 1)
        graph['rshoulder_edge_scores'] = graph['rshoulder_edge_scores'].view(-1, 1)
        graph['relbow_edge_scores'] = graph['relbow_edge_scores'].view(-1, 1)
        graph['rwrist_edge_scores'] = graph['rwrist_edge_scores'].view(-1, 1)
        graph['rhip_edge_scores'] = graph['rhip_edge_scores'].view(-1, 1)
        graph['rknee_edge_scores'] = graph['rknee_edge_scores'].view(-1, 1)
        graph['rankle_edge_scores'] = graph['rankle_edge_scores'].view(-1, 1)

        graph['neck_edge_valid'] = graph['neck_edge_valid'].view(-1, 1)
        graph['nose_edge_valid'] = graph['nose_edge_valid'].view(-1, 1)
        graph['hip_edge_valid'] = graph['hip_edge_valid'].view(-1, 1)
        graph['lshoulder_edge_valid'] = graph['lshoulder_edge_valid'].view(-1, 1)
        graph['lelbow_edge_valid'] = graph['lelbow_edge_valid'].view(-1, 1)
        graph['lwrist_edge_valid'] = graph['lwrist_edge_valid'].view(-1, 1)
        graph['lhip_edge_valid'] = graph['lhip_edge_valid'].view(-1, 1)
        graph['lknee_edge_valid'] = graph['lknee_edge_valid'].view(-1, 1)
        graph['lankle_edge_valid'] = graph['lankle_edge_valid'].view(-1, 1)
        graph['rshoulder_edge_valid'] = graph['rshoulder_edge_valid'].view(-1, 1)
        graph['relbow_edge_valid'] = graph['relbow_edge_valid'].view(-1, 1)
        graph['rwrist_edge_valid'] = graph['rwrist_edge_valid'].view(-1, 1)
        graph['rhip_edge_valid'] = graph['rhip_edge_valid'].view(-1, 1)
        graph['rknee_edge_valid'] = graph['rknee_edge_valid'].view(-1, 1)
        graph['rankle_edge_valid'] = graph['rankle_edge_valid'].view(-1, 1)

        graph['edge_labels'] = graph['edge_labels'].view(-1, 1)

        batch_size, num_cameras, num_channels, height, width = feature_maps.shape
        feature_maps = feature_maps.view(-1, num_channels, height, width)

        # multiview_centers = graph['multiview_centers'].view(batch_size * num_cameras, 1, -1, 3)
        # multiview_centers_norm = multiview_centers[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        # features = F.grid_sample(feature_maps, multiview_centers_norm, align_corners=True)[:, :, 0]
        # node_features = features.transpose(-1, -2).contiguous().view(-1, num_channels)
        # graph['node_features'] = node_features

        multiview_necks = graph['multiview_necks'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_necks_norm = multiview_necks[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        neck_features = F.grid_sample(feature_maps, multiview_necks_norm, align_corners=True)[:, :, 0]
        neck_node_features = neck_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['neck_node_features'] = neck_node_features

        multiview_noses = graph['multiview_noses'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_noses_norm = multiview_noses[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        nose_features = F.grid_sample(feature_maps, multiview_noses_norm, align_corners=True)[:, :, 0]
        nose_node_features = nose_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['nose_node_features'] = nose_node_features

        multiview_hips = graph['multiview_hips'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_hips_norm = multiview_hips[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        hip_features = F.grid_sample(feature_maps, multiview_hips_norm, align_corners=True)[:, :, 0]
        hip_node_features = hip_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['hip_node_features'] = hip_node_features

        multiview_lshoulders = graph['multiview_lshoulders'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_lshoulders_norm = multiview_lshoulders[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        lshoulder_features = F.grid_sample(feature_maps, multiview_lshoulders_norm, align_corners=True)[:, :, 0]
        lshoulder_node_features = lshoulder_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['lshoulder_node_features'] = lshoulder_node_features

        multiview_lelbows = graph['multiview_lelbows'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_lelbows_norm = multiview_lelbows[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        lelbow_features = F.grid_sample(feature_maps, multiview_lelbows_norm, align_corners=True)[:, :, 0]
        lelbow_node_features = lelbow_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['lelbow_node_features'] = lelbow_node_features

        multiview_lwrists = graph['multiview_lwrists'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_lwrists_norm = multiview_lwrists[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        lwrist_features = F.grid_sample(feature_maps, multiview_lwrists_norm, align_corners=True)[:, :, 0]
        lwrist_node_features = lwrist_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['lwrist_node_features'] = lwrist_node_features

        multiview_lhips = graph['multiview_lhips'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_lhips_norm = multiview_lhips[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        lhip_features = F.grid_sample(feature_maps, multiview_lhips_norm, align_corners=True)[:, :, 0]
        lhip_node_features = lhip_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['lhip_node_features'] = lhip_node_features

        multiview_lknees = graph['multiview_lknees'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_lknees_norm = multiview_lknees[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        lknee_features = F.grid_sample(feature_maps, multiview_lknees_norm, align_corners=True)[:, :, 0]
        lknee_node_features = lknee_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['lknee_node_features'] = lknee_node_features

        multiview_lankles = graph['multiview_lankles'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_lankles_norm = multiview_lankles[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        lankle_features = F.grid_sample(feature_maps, multiview_lankles_norm, align_corners=True)[:, :, 0]
        lankle_node_features = lankle_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['lankle_node_features'] = lankle_node_features

        multiview_rshoulders = graph['multiview_rshoulders'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_rshoulders_norm = multiview_rshoulders[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        rshoulder_features = F.grid_sample(feature_maps, multiview_rshoulders_norm, align_corners=True)[:, :, 0]
        rshoulder_node_features = rshoulder_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['rshoulder_node_features'] = rshoulder_node_features

        multiview_relbows = graph['multiview_relbows'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_relbows_norm = multiview_relbows[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        relbow_features = F.grid_sample(feature_maps, multiview_relbows_norm, align_corners=True)[:, :, 0]
        relbow_node_features = relbow_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['relbow_node_features'] = relbow_node_features

        multiview_rwrists = graph['multiview_rwrists'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_rwrists_norm = multiview_rwrists[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        rwrist_features = F.grid_sample(feature_maps, multiview_rwrists_norm, align_corners=True)[:, :, 0]
        rwrist_node_features = rwrist_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['rwrist_node_features'] = rwrist_node_features

        multiview_rhips = graph['multiview_rhips'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_rhips_norm = multiview_rhips[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        rhip_features = F.grid_sample(feature_maps, multiview_rhips_norm, align_corners=True)[:, :, 0]
        rhip_node_features = rhip_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['rhip_node_features'] = rhip_node_features

        multiview_rknees = graph['multiview_rknees'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_rknees_norm = multiview_rknees[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        rknee_features = F.grid_sample(feature_maps, multiview_rknees_norm, align_corners=True)[:, :, 0]
        rknee_node_features = rknee_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['rknee_node_features'] = rknee_node_features

        multiview_rankles = graph['multiview_rankles'].view(batch_size * num_cameras, 1, -1, 3)
        multiview_rankles_norm = multiview_rankles[..., :2] / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        rankle_features = F.grid_sample(feature_maps, multiview_rankles_norm, align_corners=True)[:, :, 0]
        rankle_node_features = rankle_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['rankle_node_features'] = rankle_node_features

        return graph

    def forward_test(self, feature_maps, img_metas):
        device = feature_maps.device
        centers_pixel, center_values = self.top_k(feature_maps[:, :, self.cfg_2d['center_channel']])

        neck_pixel, neck_values = self.top_k(feature_maps[:, :, 0])
        nose_pixel, nose_values = self.top_k(feature_maps[:, :, 1])
        hip_pixel, hip_values = self.top_k(feature_maps[:, :, 2])
        lshoulder_pixel, lshoulder_values = self.top_k(feature_maps[:, :, 3])
        lelbow_pixel, lelbow_values = self.top_k(feature_maps[:, :, 4])
        lwrist_pixel, lwrist_values = self.top_k(feature_maps[:, :, 5])
        lhip_pixel, lhip_values = self.top_k(feature_maps[:, :, 6])
        lknee_pixel, lknee_values = self.top_k(feature_maps[:, :, 7])
        lankle_pixel, lankle_values = self.top_k(feature_maps[:, :, 8])
        rshoulder_pixel, rshoulder_values = self.top_k(feature_maps[:, :, 9])
        relbow_pixel, relbow_values = self.top_k(feature_maps[:, :, 10])
        rwrist_pixel, rwrist_values = self.top_k(feature_maps[:, :, 11])
        rhip_pixel, rhip_values = self.top_k(feature_maps[:, :, 12])
        rknee_pixel, rknee_values = self.top_k(feature_maps[:, :, 13])
        rankle_pixel, rankle_values = self.top_k(feature_maps[:, :, 14])


        center_values = torch.where(center_values < self.cfg_2d['val_threshold'],torch.zeros_like(center_values), center_values)

        neck_values = torch.where(neck_values < self.cfg_2d['val_threshold'],torch.zeros_like(neck_values), neck_values)
        nose_values = torch.where(nose_values < self.cfg_2d['val_threshold'],torch.zeros_like(nose_values), nose_values)
        hip_values = torch.where(hip_values < self.cfg_2d['val_threshold'],torch.zeros_like(hip_values), hip_values)
        lshoulder_values = torch.where(lshoulder_values < self.cfg_2d['val_threshold'],torch.zeros_like(lshoulder_values), lshoulder_values)
        lelbow_values = torch.where(lelbow_values < self.cfg_2d['val_threshold'],torch.zeros_like(lelbow_values), lelbow_values)
        lwrist_values = torch.where(lwrist_values < self.cfg_2d['val_threshold'],torch.zeros_like(lwrist_values), lwrist_values)
        lhip_values = torch.where(lhip_values < self.cfg_2d['val_threshold'],torch.zeros_like(lhip_values), lhip_values)
        lknee_values = torch.where(lknee_values < self.cfg_2d['val_threshold'],torch.zeros_like(lknee_values), lknee_values)
        lankle_values = torch.where(lankle_values < self.cfg_2d['val_threshold'],torch.zeros_like(lankle_values), lankle_values)
        rshoulder_values = torch.where(rshoulder_values < self.cfg_2d['val_threshold'],torch.zeros_like(rshoulder_values), rshoulder_values)
        relbow_values = torch.where(relbow_values < self.cfg_2d['val_threshold'],torch.zeros_like(relbow_values), relbow_values)
        rwrist_values = torch.where(rwrist_values < self.cfg_2d['val_threshold'],torch.zeros_like(rwrist_values), rwrist_values)
        rhip_values = torch.where(rhip_values < self.cfg_2d['val_threshold'],torch.zeros_like(rhip_values), rhip_values)
        rknee_values = torch.where(rknee_values < self.cfg_2d['val_threshold'],torch.zeros_like(rknee_values), rknee_values)
        rankle_values = torch.where(rankle_values < self.cfg_2d['val_threshold'],torch.zeros_like(rankle_values), rankle_values)

        batch_size, num_cameras, num_persons = center_values.shape
        num_channels, height, width = feature_maps.shape[2:]
        feature_maps = feature_maps.view(-1, num_channels, height, width) #10, 527, 128, 240
        
        
        multiview_centers = centers_pixel.view(batch_size * num_cameras,1, -1, 2)

        multiview_necks = neck_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_noses = nose_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_hips = hip_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_lshoulders = lshoulder_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_lelbows = lelbow_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_lwrists = lwrist_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_lhips = lhip_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_lknees = lknee_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_lankles = lankle_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_rshoulders = rshoulder_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_relbows = relbow_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_rwrists = rwrist_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_rhips = rhip_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_rknees = rknee_pixel.view(batch_size * num_cameras,1, -1, 2)
        multiview_rankles = rankle_pixel.view(batch_size * num_cameras,1, -1, 2)


        multiview_centers_norm = multiview_centers / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0

        multiview_necks_norm = multiview_necks / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_noses_norm = multiview_noses / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_hips_norm = multiview_hips / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_lshoulders_norm = multiview_lshoulders / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_lelbows_norm = multiview_lelbows / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_lwrists_norm = multiview_lwrists / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_lhips_norm = multiview_lhips / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_lknees_norm = multiview_lknees / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_lankles_norm = multiview_lankles / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_rshoulders_norm = multiview_rshoulders / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_relbows_norm = multiview_relbows / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_rwrists_norm = multiview_rwrists / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_rhips_norm = multiview_rhips / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_rknees_norm = multiview_rknees / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        multiview_rankles_norm = multiview_rankles / (self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0


        features = F.grid_sample(feature_maps, multiview_centers_norm, align_corners=True)[:, :, 0]

        neck_features = F.grid_sample(feature_maps, multiview_necks_norm, align_corners=True)[:, :, 0]
        nose_features = F.grid_sample(feature_maps, multiview_noses_norm, align_corners=True)[:, :, 0]
        hip_features = F.grid_sample(feature_maps, multiview_hips_norm, align_corners=True)[:, :, 0]
        lshoulder_features = F.grid_sample(feature_maps, multiview_lshoulders_norm, align_corners=True)[:, :, 0]
        lelbow_features = F.grid_sample(feature_maps, multiview_lelbows_norm, align_corners=True)[:, :, 0]
        lwrist_features = F.grid_sample(feature_maps, multiview_lwrists_norm, align_corners=True)[:, :, 0]
        lhip_features = F.grid_sample(feature_maps, multiview_lhips_norm, align_corners=True)[:, :, 0]
        lknee_features = F.grid_sample(feature_maps, multiview_lknees_norm, align_corners=True)[:, :, 0]
        lankle_features = F.grid_sample(feature_maps, multiview_lankles_norm, align_corners=True)[:, :, 0]
        rshoulder_features = F.grid_sample(feature_maps, multiview_rshoulders_norm, align_corners=True)[:, :, 0]
        relbow_features = F.grid_sample(feature_maps, multiview_relbows_norm, align_corners=True)[:, :, 0]
        rwrist_features = F.grid_sample(feature_maps, multiview_rwrists_norm, align_corners=True)[:, :, 0]
        rhip_features = F.grid_sample(feature_maps, multiview_rhips_norm, align_corners=True)[:, :, 0]
        rknee_features = F.grid_sample(feature_maps, multiview_rknees_norm, align_corners=True)[:, :, 0]
        rankle_features = F.grid_sample(feature_maps, multiview_rankles_norm, align_corners=True)[:, :, 0]


        node_features = features.transpose(-1, -2).contiguous().view(-1, num_channels)

        neck_node_features = neck_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        nose_node_features = nose_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        hip_node_features = hip_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        lshoulder_node_features = lshoulder_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        lelbow_node_features = lelbow_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        lwrist_node_features = lwrist_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        lhip_node_features = lhip_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        lknee_node_features = lknee_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        lankle_node_features = lankle_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        rshoulder_node_features = rshoulder_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        relbow_node_features = relbow_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        rwrist_node_features = rwrist_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        rhip_node_features = rhip_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        rknee_node_features = rknee_features.transpose(-1, -2).contiguous().view(-1, num_channels)
        rankle_node_features = rankle_features.transpose(-1, -2).contiguous().view(-1, num_channels)


        camera_list = [SimpleCameraTorch(param=camera_param.copy(), device=device) for camera_param in img_metas[0]['camera']]

        monocular_geometries = [MonocularGeometry(transform_preds_torch(centers_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]

        neck_monocular_geometries = [MonocularGeometry(transform_preds_torch(neck_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        nose_monocular_geometries = [MonocularGeometry(transform_preds_torch(nose_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        hip_monocular_geometries = [MonocularGeometry(transform_preds_torch(hip_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        lshoulder_monocular_geometries = [MonocularGeometry(transform_preds_torch(lshoulder_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        lelbow_monocular_geometries = [MonocularGeometry(transform_preds_torch(lelbow_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        lwrist_monocular_geometries = [MonocularGeometry(transform_preds_torch(lwrist_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        lhip_monocular_geometries = [MonocularGeometry(transform_preds_torch(lhip_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        lknee_monocular_geometries = [MonocularGeometry(transform_preds_torch(lknee_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        lankle_monocular_geometries = [MonocularGeometry(transform_preds_torch(lankle_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        rshoulder_monocular_geometries = [MonocularGeometry(transform_preds_torch(rshoulder_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        relbow_monocular_geometries = [MonocularGeometry(transform_preds_torch(relbow_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        rwrist_monocular_geometries = [MonocularGeometry(transform_preds_torch(rwrist_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        rhip_monocular_geometries = [MonocularGeometry(transform_preds_torch(rhip_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        rknee_monocular_geometries = [MonocularGeometry(transform_preds_torch(rknee_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
        rankle_monocular_geometries = [MonocularGeometry(transform_preds_torch(rankle_pixel[:, i].contiguous().view(-1, 2), img_metas[0]['center'][0], img_metas[0]['scale'][0] / 200.0, self.feature_map_size),
        camera_list[i]) for i in range(num_cameras)]
       
        edge_indices = []

        # stereo_geometries = []
        neck_stereo_geometries = []
        nose_stereo_geometries = []
        hip_stereo_geometries = []
        lshoulder_stereo_geometries = []
        lelbow_stereo_geometries = []
        lwrist_stereo_geometries = []
        lhip_stereo_geometries = []
        lknee_stereo_geometries = []
        lankle_stereo_geometries = []
        rshoulder_stereo_geometries = []
        relbow_stereo_geometries = []
        rwrist_stereo_geometries = []
        rhip_stereo_geometries = []
        rknee_stereo_geometries = []
        rankle_stereo_geometries = []

        # edge_valid = []
        neck_edge_valid = []
        nose_edge_valid = []
        hip_edge_valid = []
        lshoulder_edge_valid = []
        lelbow_edge_valid = []
        lwrist_edge_valid = []
        lhip_edge_valid = []
        lknee_edge_valid = []
        lankle_edge_valid = []
        rshoulder_edge_valid = []
        relbow_edge_valid = []
        rwrist_edge_valid = []
        rhip_edge_valid = []
        rknee_edge_valid = []
        rankle_edge_valid = []

        for camera_pair in self.camera_pairs: #camera_pair ; 2 self.cam 10,2
            # nodes_valid = center_values[:, camera_pair] > 0  # batch_size x 2 x num_persons

            neck_nodes_valid = neck_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            nose_nodes_valid = nose_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            hip_nodes_valid = hip_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            lshoulder_nodes_valid = lshoulder_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            lelbow_nodes_valid = lelbow_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            lwrist_nodes_valid = lwrist_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            lhip_nodes_valid = lhip_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            lknee_nodes_valid = lknee_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            lankle_nodes_valid = lankle_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            rshoulder_nodes_valid = rshoulder_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            relbow_nodes_valid = relbow_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            rwrist_nodes_valid = rwrist_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            rhip_nodes_valid = rhip_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            rknee_nodes_valid = rknee_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            rankle_nodes_valid = rankle_values[:, camera_pair] > 0  # batch_size x 2 x num_persons

            # nodes_valid_src, nodes_valid_tar = nodes_valid[:, 0], nodes_valid[:, 1]

            neck_nodes_valid_src, neck_nodes_valid_tar = neck_nodes_valid[:, 0], neck_nodes_valid[:, 1]
            nose_nodes_valid_src, nose_nodes_valid_tar = nose_nodes_valid[:, 0], nose_nodes_valid[:, 1]
            hip_nodes_valid_src, hip_nodes_valid_tar = hip_nodes_valid[:, 0], hip_nodes_valid[:, 1]
            lshoulder_nodes_valid_src, lshoulder_nodes_valid_tar = lshoulder_nodes_valid[:, 0], lshoulder_nodes_valid[:, 1]
            lelbow_nodes_valid_src, lelbow_nodes_valid_tar = lelbow_nodes_valid[:, 0], lelbow_nodes_valid[:, 1]
            lwrist_nodes_valid_src, lwrist_nodes_valid_tar = lwrist_nodes_valid[:, 0], lwrist_nodes_valid[:, 1]
            lhip_nodes_valid_src, lhip_nodes_valid_tar = lhip_nodes_valid[:, 0], lhip_nodes_valid[:, 1]
            lknee_nodes_valid_src, lknee_nodes_valid_tar = lknee_nodes_valid[:, 0], lknee_nodes_valid[:, 1]
            lankle_nodes_valid_src, lankle_nodes_valid_tar = lankle_nodes_valid[:, 0], lankle_nodes_valid[:, 1]
            rshoulder_nodes_valid_src, rshoulder_nodes_valid_tar = rshoulder_nodes_valid[:, 0], rshoulder_nodes_valid[:, 1]
            relbow_nodes_valid_src, relbow_nodes_valid_tar = relbow_nodes_valid[:, 0], relbow_nodes_valid[:, 1]
            rwrist_nodes_valid_src, rwrist_nodes_valid_tar = rwrist_nodes_valid[:, 0], rwrist_nodes_valid[:, 1]
            rhip_nodes_valid_src, rhip_nodes_valid_tar = rhip_nodes_valid[:, 0], rhip_nodes_valid[:, 1]
            rknee_nodes_valid_src, rknee_nodes_valid_tar = rknee_nodes_valid[:, 0], rknee_nodes_valid[:, 1]
            rankle_nodes_valid_src, rankle_nodes_valid_tar = rankle_nodes_valid[:, 0], rankle_nodes_valid[:, 1]


            masks = torch.stack([self.person_pairs + i * num_persons for i in range(batch_size)], dim=0)  # batch_size x E x 2
            masks_src, masks_tar = masks[..., 0], masks[..., 1]
            # tik = time()
            # stereo_geometry = StereoGeometry(monocular_geometries[camera_pair[0].item()], monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))

            neck_stereo_geometry = StereoGeometry(neck_monocular_geometries[camera_pair[0].item()], neck_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            nose_stereo_geometry = StereoGeometry(nose_monocular_geometries[camera_pair[0].item()], nose_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            hip_stereo_geometry = StereoGeometry(hip_monocular_geometries[camera_pair[0].item()], hip_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            lshoulder_stereo_geometry = StereoGeometry(lshoulder_monocular_geometries[camera_pair[0].item()], lshoulder_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            lelbow_stereo_geometry = StereoGeometry(lelbow_monocular_geometries[camera_pair[0].item()], lelbow_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            lwrist_stereo_geometry = StereoGeometry(lwrist_monocular_geometries[camera_pair[0].item()], lwrist_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            lhip_stereo_geometry = StereoGeometry(lhip_monocular_geometries[camera_pair[0].item()], lhip_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            lknee_stereo_geometry = StereoGeometry(lknee_monocular_geometries[camera_pair[0].item()], lknee_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            lankle_stereo_geometry = StereoGeometry(lankle_monocular_geometries[camera_pair[0].item()], lankle_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            rshoulder_stereo_geometry = StereoGeometry(rshoulder_monocular_geometries[camera_pair[0].item()], rshoulder_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            relbow_stereo_geometry = StereoGeometry(relbow_monocular_geometries[camera_pair[0].item()], relbow_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            rwrist_stereo_geometry = StereoGeometry(rwrist_monocular_geometries[camera_pair[0].item()], rwrist_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            rhip_stereo_geometry = StereoGeometry(rhip_monocular_geometries[camera_pair[0].item()], rhip_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            rknee_stereo_geometry = StereoGeometry(rknee_monocular_geometries[camera_pair[0].item()], rknee_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))
            rankle_stereo_geometry = StereoGeometry(rankle_monocular_geometries[camera_pair[0].item()], rankle_monocular_geometries[camera_pair[1].item()], masks_src.view(-1), masks_tar.view(-1))

            
            # edge_valid_ = nodes_valid_src[:, masks_src[0]] * nodes_valid_tar[:, masks_tar[0]]

            neck_edge_valid_ = neck_nodes_valid_src[:, masks_src[0]] * neck_nodes_valid_tar[:, masks_tar[0]]
            nose_edge_valid_ = nose_nodes_valid_src[:, masks_src[0]] * nose_nodes_valid_tar[:, masks_tar[0]]
            hip_edge_valid_ = hip_nodes_valid_src[:, masks_src[0]] * hip_nodes_valid_tar[:, masks_tar[0]]
            lshoulder_edge_valid_ = lshoulder_nodes_valid_src[:, masks_src[0]] * lshoulder_nodes_valid_tar[:, masks_tar[0]]
            lelbow_edge_valid_ = lelbow_nodes_valid_src[:, masks_src[0]] * lelbow_nodes_valid_tar[:, masks_tar[0]]
            lwrist_edge_valid_ = lwrist_nodes_valid_src[:, masks_src[0]] * lwrist_nodes_valid_tar[:, masks_tar[0]]
            lhip_edge_valid_ = lhip_nodes_valid_src[:, masks_src[0]] * lhip_nodes_valid_tar[:, masks_tar[0]]
            lknee_edge_valid_ = lknee_nodes_valid_src[:, masks_src[0]] * lknee_nodes_valid_tar[:, masks_tar[0]]
            lankle_edge_valid_ = lankle_nodes_valid_src[:, masks_src[0]] * lankle_nodes_valid_tar[:, masks_tar[0]]
            rshoulder_edge_valid_ = rshoulder_nodes_valid_src[:, masks_src[0]] * rshoulder_nodes_valid_tar[:, masks_tar[0]]
            relbow_edge_valid_ = relbow_nodes_valid_src[:, masks_src[0]] * relbow_nodes_valid_tar[:, masks_tar[0]]
            rwrist_edge_valid_ = rwrist_nodes_valid_src[:, masks_src[0]] * rwrist_nodes_valid_tar[:, masks_tar[0]]
            rhip_edge_valid_ = rhip_nodes_valid_src[:, masks_src[0]] * rhip_nodes_valid_tar[:, masks_tar[0]]
            rknee_edge_valid_ = rknee_nodes_valid_src[:, masks_src[0]] * rknee_nodes_valid_tar[:, masks_tar[0]]
            rankle_edge_valid_ = rankle_nodes_valid_src[:, masks_src[0]] * rankle_nodes_valid_tar[:, masks_tar[0]]



            edge_index = torch.cat([self.person_pairs + camera_pair[None] * num_persons + i * num_persons * num_cameras for i in range(batch_size)], dim=0)  # batch_size * E x 2
            edge_indices.append(edge_index)

            # edge_valid.append(edge_valid_.contiguous().view(-1))
            neck_edge_valid.append(neck_edge_valid_.contiguous().view(-1))
            nose_edge_valid.append(nose_edge_valid_.contiguous().view(-1))
            hip_edge_valid.append(hip_edge_valid_.contiguous().view(-1))
            lshoulder_edge_valid.append(lshoulder_edge_valid_.contiguous().view(-1))
            lelbow_edge_valid.append(lelbow_edge_valid_.contiguous().view(-1))
            lwrist_edge_valid.append(lwrist_edge_valid_.contiguous().view(-1))
            lhip_edge_valid.append(lhip_edge_valid_.contiguous().view(-1))
            lknee_edge_valid.append(lknee_edge_valid_.contiguous().view(-1))
            lankle_edge_valid.append(lankle_edge_valid_.contiguous().view(-1))
            rshoulder_edge_valid.append(rshoulder_edge_valid_.contiguous().view(-1))
            relbow_edge_valid.append(relbow_edge_valid_.contiguous().view(-1))
            rwrist_edge_valid.append(rwrist_edge_valid_.contiguous().view(-1))
            rhip_edge_valid.append(rhip_edge_valid_.contiguous().view(-1))
            rknee_edge_valid.append(rknee_edge_valid_.contiguous().view(-1))
            rankle_edge_valid.append(rankle_edge_valid_.contiguous().view(-1))

            stereo_geometries.append(stereo_geometry)

        edge_scores_1to2 = torch.cat([torch.exp(-self.cfg_2d['dist_coef'] * stereo_geometry.distance_1to2) for stereo_geometry in stereo_geometries], dim=0)  # num_pairs * batch_size * E
        edge_scores_2to1 = torch.cat([torch.exp(-self.cfg_2d['dist_coef'] * stereo_geometry.distance_2to1) for stereo_geometry in stereo_geometries], dim=0)
        edge_indices = torch.cat(edge_indices, dim=0).long()  # num_pairs * batch_size * E x 2
        edge_valid = torch.cat(edge_valid, dim=0)

        input_edge_indices = torch.cat([edge_indices[edge_valid],
                                        edge_indices[edge_valid].flip([-1])], dim=0)
        input_edge_scores = torch.cat([edge_scores_1to2[edge_valid],
                                       edge_scores_2to1[edge_valid]], dim=0)
        edge_preds = torch.zeros_like(edge_scores_1to2)
        _, preds = self.match_gcn(node_features, input_edge_indices, input_edge_scores[:, None])
        preds = preds.sigmoid()
        edge_preds[edge_valid] = preds.view(2, -1).mean(0)

        stereo_reconstructions = torch.stack([stereo_geometry.reconstructions
                                              for stereo_geometry in stereo_geometries], dim=0)

        match_results = dict(edge_preds=edge_preds,  # num_pairs * batch_size * E
                             edge_valid=edge_valid,
                             node_valid=center_values > 0,  # batch_size x num_cameras x num_persons
                             edge_indices=edge_indices,
                             batch_size=batch_size,
                             num_cameras=num_cameras,
                             num_persons=num_persons,
                             stereo_reconstructions=stereo_reconstructions,  # num_pairs x batch_size * E x 3
                             monocular_geometries=monocular_geometries)
          # num_cameras x batch_size*num_persons
        coarse_candidates, pixel_ray_candidates = self.get_coarse_candidates(match_results)
        all_coarse_candidates, all_pixel_ray_candidates = self.get_all_candidates(match_results)
        center_candidates = self.generate_center_candidates(coarse_candidates, pixel_ray_candidates)
        all_center_candidates = self.generate_center_candidates(all_coarse_candidates, all_pixel_ray_candidates)

        batch0_all_coarse_candidates =  all_coarse_candidates[0]
        batch0_coarse_candidates = coarse_candidates[0]
        # img_3d = imshow_multiview_joints_3d(
        #         batch0_all_coarse_candidates.cpu().numpy(),
        #         space_size=[8000, 8000, 2000],
        #         space_center=[0, -500, 800])
        # # torch.stack(all_coarses_candidates).cpu()
            
        # mmcv.image.imwrite(img_3d,os.path.join('vis_result', 'vis_3d', 'all_joints.jpg'))
        return center_candidates

    def get_coarse_candidates(self, match_results):
        batch_size = match_results['batch_size']
        num_pairs = len(self.camera_pairs)
        edge_preds = match_results['edge_preds'].view(num_pairs, batch_size, -1)  # num_pairs * batch_size * E
        edge_valid = match_results['edge_valid'].view(num_pairs, batch_size, -1)
        node_valid = match_results['node_valid']
        edge_indices = match_results['edge_indices'].view(num_pairs, batch_size, -1, 2)
        stereo_reconstructions = match_results['stereo_reconstructions'].view(num_pairs, batch_size, -1, 3) #10,2,100,3
        monocular_geometries = match_results['monocular_geometries']
        camera_centers = torch.stack([mono_geo.camera_center.T
                                      for mono_geo in monocular_geometries], dim=0)  # num_cameras x 1 x 3
        ray_directions = torch.stack([mono_geo.ray_direction.T.view(batch_size, -1, 3)
                                      for mono_geo in monocular_geometries], dim=0) # 5,2,10,3
        # num_cameras x batch_size x num_persons x 3

        coarse_candidates = []
        pixel_ray_candidates = []
        unmatched_nodes = node_valid.view(-1)
        for i in range(batch_size):
            edge_preds_i = edge_preds[:, i].contiguous().view(-1)
            edge_valid_i = edge_valid[:, i].contiguous().view(-1)
            edge_indices_i = edge_indices[:, i].contiguous().view(-1, 2)
            src_indices, tar_indices = edge_indices_i[:, 0], edge_indices_i[:, 1]
            matched = edge_valid_i * (edge_preds_i > self.match_threshold)

            unmatched_nodes[src_indices[matched]] = 0
            unmatched_nodes[tar_indices[matched]] = 0

            unmatched_nodes_i = unmatched_nodes.view(batch_size, -1)[i]

            reconstructions_i = stereo_reconstructions[:, i].contiguous().view(-1, 3)
            ray_directions_i = ray_directions[:, i]
            camera_centers_i = torch.zeros_like(ray_directions_i) + camera_centers
            pixel_rays_i = torch.cat([camera_centers_i,ray_directions_i], dim=-1).view(-1, 6)

            coarse_candidates.append(reconstructions_i[matched])
            pixel_ray_candidates.append(pixel_rays_i[unmatched_nodes_i])

        return coarse_candidates, pixel_ray_candidates
        

    def generate_center_candidates(self, coarse_candidates, pixel_ray_candidates):
        center_candidates = []
        batch_size = len(coarse_candidates)
        for i in range(batch_size):
            samples_to_query = self.grid_samples[:, 0].logical_not()
            coarse_candidates_i = coarse_candidates[i]
            pixel_ray_candidates_i = pixel_ray_candidates[i]

            if len(coarse_candidates_i) > 0:
                dist_to_candidates_i, _ = torch.norm(self.grid_samples[:, None] - coarse_candidates_i[None], dim=-1).min(-1)
                samples_to_query[dist_to_candidates_i < self.cfg_3d['dist_threshold']] = 1

            if len(pixel_ray_candidates_i) > 0:
                s = torch.cross(pixel_ray_candidates_i[None, :, :3]
                                - self.grid_samples[:, None],
                                pixel_ray_candidates_i[None, :, :3]
                                + pixel_ray_candidates_i[None, :, 3:]
                                - self.grid_samples[:, None], dim=-1).norm(dim=-1)  # num_bins x num_rays
                dist_to_rays_i = s / (pixel_ray_candidates_i[None, :, 3:].norm(dim=-1) + 1e-12)
                dist_to_rays_i = dist_to_rays_i.min(-1)[0]
                samples_to_query[dist_to_rays_i < self.cfg_3d['dist_threshold']] = 1
            if samples_to_query.sum().int() == 0:
                samples_to_query = self._get_sparse_candidates(samples_to_query)
            center_candidates.append(samples_to_query)

        return center_candidates
    

    def get_all_candidates(self, match_results):
        batch_size = match_results['batch_size']
        num_pairs = len(self.camera_pairs)
        edge_preds = match_results['edge_preds'].view(num_pairs, batch_size, -1)  # num_pairs * batch_size * E
        edge_valid = match_results['edge_valid'].view(num_pairs, batch_size, -1)
        node_valid = match_results['node_valid']
        edge_indices = match_results['edge_indices'].view(num_pairs, batch_size, -1, 2)
        stereo_reconstructions = match_results['stereo_reconstructions'].view(num_pairs, batch_size, -1, 3)
        monocular_geometries = match_results['monocular_geometries']
        camera_centers = torch.stack([mono_geo.camera_center.T
                                      for mono_geo in monocular_geometries], dim=0)  # num_cameras x 1 x 3
        ray_directions = torch.stack([mono_geo.ray_direction.T.view(batch_size, -1, 3)
                                      for mono_geo in monocular_geometries], dim=0)
        # num_cameras x batch_size x num_persons x 3

        coarse_candidates = []
        pixel_ray_candidates = []
        unmatched_nodes = node_valid.view(-1)
        for i in range(batch_size):
            edge_preds_i = edge_preds[:, i].contiguous().view(-1)
            edge_valid_i = edge_valid[:, i].contiguous().view(-1)
            edge_indices_i = edge_indices[:, i].contiguous().view(-1, 2)
            src_indices, tar_indices = edge_indices_i[:, 0], edge_indices_i[:, 1]
            matched = edge_valid_i * (edge_preds_i > self.match_threshold)

            unmatched_nodes[src_indices[matched]] = 0
            unmatched_nodes[tar_indices[matched]] = 0

            unmatched_nodes_i = unmatched_nodes.view(batch_size, -1)[i]

            reconstructions_i = stereo_reconstructions[:, i].contiguous().view(-1, 3)
            ray_directions_i = ray_directions[:, i]
            camera_centers_i = torch.zeros_like(ray_directions_i) + camera_centers
            pixel_rays_i = torch.cat([camera_centers_i,
                                      ray_directions_i], dim=-1).view(-1, 6)

            coarse_candidates.append(reconstructions_i)
            pixel_ray_candidates.append(pixel_rays_i)

        return coarse_candidates, pixel_ray_candidates
    
    def generate_all_center_candidates(self, coarse_candidates, pixel_ray_candidates):
        center_candidates = []
        batch_size = len(coarse_candidates)
        for i in range(batch_size):
            samples_to_query = self.grid_samples[:, 0].logical_not()
            coarse_candidates_i = coarse_candidates[i]
            pixel_ray_candidates_i = pixel_ray_candidates[i]

            if len(coarse_candidates_i) > 0:
                dist_to_candidates_i, _ = torch.norm(self.grid_samples[:, None] - coarse_candidates_i[None], dim=-1).min(-1)
                samples_to_query[dist_to_candidates_i < self.cfg_3d['dist_threshold']] = 1

            if len(pixel_ray_candidates_i) > 0:
                s = torch.cross(pixel_ray_candidates_i[None, :, :3]
                                - self.grid_samples[:, None],
                                pixel_ray_candidates_i[None, :, :3]
                                + pixel_ray_candidates_i[None, :, 3:]
                                - self.grid_samples[:, None], dim=-1).norm(dim=-1)  # num_bins x num_rays
                dist_to_rays_i = s / (pixel_ray_candidates_i[None, :, 3:].norm(dim=-1) + 1e-12)
                dist_to_rays_i = dist_to_rays_i.min(-1)[0]
                samples_to_query[dist_to_rays_i < self.cfg_3d['dist_threshold']] = 1
            if samples_to_query.sum().int() == 0:
                samples_to_query = self._get_sparse_candidates(samples_to_query)
            center_candidates.append(samples_to_query)

        return center_candidates

