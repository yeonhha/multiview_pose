import torch
from mmpose.models.builder import POSENETS
from mmpose.models.detectors import DetectAndRegress
from mmpose.core import imshow_keypoints, imshow_multiview_keypoints_3d
from multiview_pose.models.gcn_modules import GCNS
import mmcv
import os
from mmpose.datasets.dataset_info import DatasetInfo
from mmcv import Config
from matplotlib import pyplot as plt
import numpy as np

def imshow_multiview_joints_3d(
    pose_result,
    skeleton=None,
    pose_kpt_color=None,
    pose_link_color=None,
    space_size=[8000, 8000, 2000],
    space_center=[0, -500, 800],
    kpt_score_thr=0.0,
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
    pose_kpt_color = np.array(pose_kpt_color)
    pose_kpt_color = pose_kpt_color[..., ::-1] / 255.

    for kpts in pose_result:
        # draw each point on image
        xs, ys, zs, scores = kpts.T
        valid = scores > kpt_score_thr
        ax.scatter(
            xs[valid],
            ys[valid],
            zs[valid],
            marker='o',
            color=pose_kpt_color[valid])

    # convert figure to numpy array
    fig.tight_layout()
    fig.canvas.draw()
    img_w, img_h = fig.canvas.get_width_height()
    img_vis = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(img_h, img_w, -1)
    img_vis = mmcv.rgb2bgr(img_vis)

    plt.close(fig)

    return img_vis


@POSENETS.register_module()
class GraphBasedModel(DetectAndRegress):
    def __init__(self, num_joints, pose_refiner, test_with_refine=True, freeze_keypoint_head=True,
                 *args, **kwargs):
        super(GraphBasedModel, self).__init__(*args, **kwargs)
        self.num_joints = num_joints
        if pose_refiner is not None:
            self.pose_refiner = GCNS.build(pose_refiner)
        else:
            self.pose_refiner = None
        self.test_with_refine = test_with_refine
        self.freeze_keypoint_head = freeze_keypoint_head

    def train(self, mode=True):
        """Sets the module in training mode.
        Args:
            mode (bool): whether to set training mode (``True``)
                or evaluation mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        super().train(mode)
        if mode and self.freeze_2d:
            if self.backbone is not None:
                self._freeze(self.backbone)
            if self.keypoint_head is not None and self.freeze_keypoint_head:
                self._freeze(self.keypoint_head)

        return self

    @property
    def has_keypoint_2d_loss(self):
        return (not self.freeze_2d) or (self.freeze_2d and not self.freeze_keypoint_head)

    def forward(self,
                img=None,
                img_metas=None,
                return_loss=True,
                target=None,
                mask=None,
                targets_3d=None,
                input_heatmaps=None,
                **kwargs):

        if return_loss:
            return self.forward_train(img, img_metas, target, mask,
                                      targets_3d, input_heatmaps, **kwargs)
        else:
            return self.forward_test(img, img_metas, input_heatmaps, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      target=None,
                      mask=None,
                      targets_3d=None,
                      input_heatmaps=None,
                      **kwargs):
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                feature_maps.append(self.predict_heatmap(img_)[0])

        losses = dict()
        human_candidates, human_loss = self.human_detector.forward_train(
            None, img_metas, feature_maps, targets_3d, return_preds=True, **kwargs)
        losses.update(human_loss)

        pose_pred, pose_loss = self.pose_regressor.forward_train(
            None,
            img_metas,
            feature_maps=[f[:, -self.num_joints:].detach() for f in feature_maps],
            human_candidates=human_candidates,
            return_preds=True)
        losses.update(pose_loss)
        if self.pose_refiner is not None:
            losses.update(self.pose_refiner.forward_train(pose_pred, feature_maps, img_metas))

        if self.has_keypoint_2d_loss:
            losses_2d = {}
            heatmaps_tensor = torch.cat([f[:, -self.num_joints:] for f in feature_maps], dim=0)
            targets_tensor = torch.cat([t[0] for t in target], dim=0)
            masks_tensor = torch.cat([m[0] for m in mask], dim=0)
            losses_2d_ = self.keypoint_head.get_loss([heatmaps_tensor],
                                                     [targets_tensor], [masks_tensor])
            for k, v in losses_2d_.items():
                losses_2d[k + '_2d'] = v
            losses.update(losses_2d)

        return losses

    def forward_test(
        self,
        img,
        img_metas,
        input_heatmaps=None,
        **kwargs
    ):
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                feature_maps.append(self.predict_heatmap(img_)[0])

        human_candidates = self.human_detector.forward_test(None, img_metas, feature_maps, **kwargs)

        human_poses = self.pose_regressor(
            None,
            img_metas,
            return_loss=False,
            feature_maps=[f[:, -self.num_joints:] for f in feature_maps],
            human_candidates=human_candidates)
        
        if self.pose_refiner is not None and self.test_with_refine:
            human_poses = self.pose_refiner.forward_test(human_poses, feature_maps, img_metas)

        result = {}
        result['pose_3d'] = human_poses.cpu().numpy()
        result['human_detection_3d'] = human_candidates.cpu().numpy()
        result['sample_id'] = [img_meta['sample_id'] for img_meta in img_metas]

        # ##전체 pose visualization
        # pose_3d = result['pose_3d']
        # sample_id = result['sample_id']
        # batch_size = pose_3d.shape[0]

        # for i in range(batch_size):
        #     img_meta = img_metas[i]
        #     num_cameras = len(img_meta['camera'])
        #     pose_3d_i = pose_3d[i]
        #     pose_3d_i = pose_3d_i[pose_3d_i[:, 0, 3] >= 0] ##max person이 10명이라서 10, 15, 5 검출 되었으나 8명은 존재하지 않아서 삭제 

        #     num_persons, num_keypoints, _ = pose_3d_i.shape
        #     pose_3d_list = [p[..., [0, 1, 2, 4]] for p in pose_3d_i] if num_persons > 0 else []
            
        #     cfg = Config.fromfile('configs/_base_/datasets/panoptic_body3d.py')
        #     dataset_info = cfg._cfg_dict['dataset_info']
        #     dataset_info = DatasetInfo(dataset_info)

        #     img_3d = imshow_multiview_keypoints_3d(
        #         pose_3d_list,
        #         skeleton=dataset_info.skeleton,
        #         pose_kpt_color=dataset_info.pose_kpt_color[:num_keypoints],
        #         pose_link_color=dataset_info.pose_link_color,
        #         space_size=self.human_detector.match_module.cfg_3d['space_size'],
        #         space_center=self.human_detector.match_module.cfg_3d['space_center'])
            
        #     mmcv.image.imwrite(img_3d,os.path.join('vis_result', 'vis_3d', f'{sample_id[i]}_3d.jpg'))

        #human candidates visualization
        # human_candidates = result['human_detection_3d']
        # sample_id = result['sample_id']
        # batch_size = human_candidates.shape[0]

        # for i in range(batch_size):
        #     img_meta = img_metas[i]
        #     num_cameras = len(img_meta['camera'])
        #     human_candidates_i = human_candidates[i]
        #     human_candidates_i = human_candidates_i[human_candidates_i[:, 3] >= 0] ##max person이 10명이라서 10, 15, 5 검출 되었으나 8명은 존재하지 않아서 삭제 

        #     num_persons, _ = human_candidates_i.shape
        #     pose_3d_list = [p[..., [0, 1, 2, 4]] for p in human_candidates_i] if num_persons > 0 else []
            
        #     cfg = Config.fromfile('configs/_base_/datasets/panoptic_body3d.py')
        #     dataset_info = cfg._cfg_dict['dataset_info']
        #     dataset_info = DatasetInfo(dataset_info)

        #     img_3d = imshow_multiview_joints_3d(
        #         pose_3d_list,
        #         skeleton=dataset_info.skeleton,
        #         pose_kpt_color=dataset_info.pose_kpt_color[2],
        #         pose_link_color=dataset_info.pose_link_color,
        #         space_size=self.human_detector.match_module.cfg_3d['space_size'],
        #         space_center=self.human_detector.match_module.cfg_3d['space_center'])
            
        #     mmcv.image.imwrite(img_3d,os.path.join('vis_result', 'vis_3d', f'{sample_id[i]}_3d_joints.jpg'))

        

        return result
