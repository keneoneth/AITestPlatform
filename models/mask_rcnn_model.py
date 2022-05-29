import tensorflow as tf
import math
import numpy as np
if __name__ == "__main__":
    from resnet_model import ResNetStruct
    import utils
else:
    from .resnet_model import ResNetStruct
    from . import utils



class MaskRCNN():

    @staticmethod
    def norm_boxes_graph(arg):
        """Converts boxes from pixel coordinates to normalized coordinates.
        boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
        shape: [..., (height, width)] in pixels

        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.

        Returns:
            [..., (y1, x1, y2, x2)] in normalized coordinates
        """
        boxes,tensor_for_shape = arg
        shape = tf.shape(tensor_for_shape)[1:3] #get H,W data
        h, w = tf.split(tf.cast(shape, tf.float32), 2)
        scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
        shift = tf.constant([0., 0., 1., 1.])
        # print("boxes shape",boxes,shape)
        # print("ans",h,w,scale,shift)
        return tf.divide(boxes - shift, scale)

    @staticmethod
    def get_compose_image_meta_len(NUM_CLASSES):
        return 1 + 3 + 3 + 4 + 1 + NUM_CLASSES

    @staticmethod
    def compose_image_meta(image_id, original_image_shape, image_shape, window, scale, active_class_ids):
        """Takes attributes of an image and puts them in one 1D array.

        image_id: An int ID of the image. Useful for debugging.
        original_image_shape: [H, W, C] before resizing or padding.
        image_shape: [H, W, C] after resizing and padding
        window: (y1, x1, y2, x2) in pixels. The area of the image where the real
                image is (excluding the padding)
        scale: The scaling factor applied to the original image (float32)
        active_class_ids: List of class_ids available in the dataset from which
            the image came. Useful if training on images from multiple datasets
            where not all classes are present in all datasets.
        """
        meta = np.array(
            [image_id] +                  # size=1
            list(original_image_shape) +  # size=3
            list(image_shape) +           # size=3
            list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
            [scale] +                     # size=1
            list(active_class_ids)        # size=num_classes
        )
        return meta

    @staticmethod
    def parse_image_meta_graph(meta):
        #TODO change it to fit aitestplatform
        """Parses a tensor that contains image attributes to its components.
        See compose_image_meta() for more details.
        meta: [batch, meta length] where meta length depends on NUM_CLASSES
        Returns a dict of the parsed tensors.
        """
        image_id = meta[:, 0]
        original_image_shape = meta[:, 1:4]
        image_shape = meta[:, 4:7]
        window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
        scale = meta[:, 11]
        active_class_ids = meta[:, 12:]
        return {
            "image_id": image_id,
            "original_image_shape": original_image_shape,
            "image_shape": image_shape,
            "window": window,
            "scale": scale,
            "active_class_ids": active_class_ids,
        }

    @staticmethod
    def norm_boxes(boxes, shape):
        """Converts boxes from pixel coordinates to normalized coordinates.
        boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
        shape: [..., (height, width)] in pixels

        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.

        Returns:
            [N, (y1, x1, y2, x2)] in normalized coordinates
        """
        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.divide((boxes - shift), scale).astype(np.float32)

    @staticmethod
    def generate_pyramid_anchors(feature_shapes,base_model_strides,anchor_stride,anchor_areas,aspect_ratios,print_debug=False):
        if print_debug:
            print("feature_shapes",feature_shapes)
            print("base_model_strides",base_model_strides)
            print("anchor_stride",anchor_stride)
            print("anchor_areas",anchor_areas)
            print("aspect_ratios",aspect_ratios)

        anchors_arr = []
        for index,area in enumerate(anchor_areas):
            f_shape = feature_shapes[index]
            f_stride = base_model_strides[index]
            
            # area = w * h
            # ratio[0]/ratio[1] = w/h
            widths = [math.sqrt(area*ratio[0]/ratio[1]) for ratio in aspect_ratios]
            heights = [area / width for width in widths]
            if print_debug:
                print('wh',widths,heights)

            shifts_y = np.arange(0, f_shape[0], anchor_stride) * f_stride
            shifts_x = np.arange(0, f_shape[1], anchor_stride) * f_stride
            # print('shifts_x,y',shifts_x.shape,shifts_y.shape,shifts_x[0],shifts_y[0])
            shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
            
            # print("new shifts_y",shifts_y.shape,shifts_y[0])
            # print("new shifts_x",shifts_x.shape,shifts_x[0])

            box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
            box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

            box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
            box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

            boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

            anchors_arr.append(boxes)

        return np.concatenate(anchors_arr, axis=0)

    def gen_anchors(self,image_shape,feature_shapes,anchor_areas,aspect_ratios,anchor_stride):
        # anchor_areas: from paper for P2,P3,P4,P5,P6
        # aspect_ratios: from paper
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if tuple(image_shape) in self._anchor_cache:
            return self._anchor_cache[tuple(image_shape)]
        # Generate Anchors
        anchors_arr = MaskRCNN.generate_pyramid_anchors(
            feature_shapes,
            MaskRCNN.BASE_MODEL_STRIDES + [MaskRCNN.P6_STRIDE],
            anchor_stride,
            anchor_areas,
            aspect_ratios)
        self._anchor_cache[tuple(image_shape)] = MaskRCNN.norm_boxes(anchors_arr,image_shape)
        
        return self._anchor_cache[tuple(image_shape)]

    def rpn_graph(self, feature_map, anchors_per_location, anchor_stride):
        """Builds the computation graph of Region Proposal Network.

        feature_map: backbone features [batch, height, width, depth]
        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                    every pixel in the feature map), or 2 (every other pixel).

        Returns:
            rpn_class_logits: [batch, H * W * anchors_per_location, 2 = fg/bg scores] Anchor classifier logits (before softmax)
            rpn_probs: [batch, H * W * anchors_per_location, 2 = fg/bg scores] Anchor classifier probabilities.
            rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                    applied to anchors.
        """
        # source: https://towardsdatascience.com/faster-r-cnn-for-object-detection-a-technical-summary-474c5b857b46

        # TODO: check if stride of 2 causes alignment issues if the feature map
        # is not even.
        # Shared convolutional base of the RPN
        print("fm",feature_map)
        # First, a 3 x 3 convolution with 512 units is applied to the backbone feature map as shown in Figure 1, 
        # to give a 512-d feature map for every location. This is followed by two sibling layers: a 1 x 1 convolution 
        # layer with 18 units for object classification, and a 1 x 1 convolution with 36 units for bounding box regression.
        # safer to use 2x units for classification?

        shared = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', strides=anchor_stride, name='rpn_conv_shared')(feature_map)

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = tf.keras.layers.Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear', name='rpn_class_raw')(shared)

        print('sp',tf.shape(x))
        # Reshape to [batch, anchors, 2]
        rpn_class_logits = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)
        print('rpn_class_logits',tf.shape(rpn_class_logits))

        # Softmax on last dimension of BG/FG.
        # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
        rpn_probs = tf.keras.layers.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = tf.keras.layers.Conv2D(anchors_per_location * 4, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred')(shared)

        # Reshape to [batch, anchors, 4]
        rpn_bbox = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

        return [rpn_class_logits, rpn_probs, rpn_bbox]

    def build_rpn_model(self, anchor_stride, anchors_per_location, depth, rpn_feature_maps):
        """Builds a Keras model of the Region Proposal Network.
        It wraps the RPN graph so it can be used multiple times with shared
        weights.

        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                    every pixel in the feature map), or 2 (every other pixel).
        depth: Depth of the backbone feature map.

        Returns a Keras Model object. The model outputs, when called, are:
        rpn_class_logits: [batch, H * W * anchors_per_location = (total number of anchors), 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location = (total number of anchors), 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be applied to anchors.
        """
        input_feature_map = tf.keras.Input(shape=[None, None, depth], name="input_rpn_feature_map")
        outputs = self.rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
        rpn =  tf.keras.Model([input_feature_map], outputs, name="rpn_model")
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [tf.keras.layers.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

        return outputs

    @staticmethod
    def trim_zeros_graph(boxes, name='trim_zeros'):
        """Often boxes are represented with matrices of shape [N, 4] and
        are padded with zeros. This removes zero boxes.

        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
        """
        non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
        boxes = tf.boolean_mask(boxes, non_zeros, name=name)
        return boxes, non_zeros

    @staticmethod
    def overlaps_graph(boxes1, boxes2):
        """Computes IoU overlaps between two sets of boxes.
        boxes1, boxes2: [N, (y1, x1, y2, x2)].
        """
        # 1. Tile boxes2 and repeat boxes1. This allows us to compare
        # every boxes1 against every boxes2 without loops.
        # TF doesn't have an equivalent to np.repeat() so simulate it
        # using tf.tile() and tf.reshape.
        b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                                [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
        b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
        # 2. Compute intersections
        b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
        b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
        y1 = tf.maximum(b1_y1, b2_y1)
        x1 = tf.maximum(b1_x1, b2_x1)
        y2 = tf.minimum(b1_y2, b2_y2)
        x2 = tf.minimum(b1_x2, b2_x2)
        intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
        # 3. Compute unions
        b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        union = b1_area + b2_area - intersection
        # 4. Compute IoU and reshape to [boxes1, boxes2]
        iou = intersection / union
        overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
        return overlaps

    @staticmethod
    def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
        """Generates detection targets for one image. Subsamples proposals and
        generates target class IDs, bounding box deltas, and masks for each.

        Inputs:
        proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
                be zero padded if there are not enough proposals.
        gt_class_ids: [MAX_GT_INSTANCES] int class IDs
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
        gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

        Returns: Target ROIs and corresponding class IDs, bounding box shifts,
        and masks.
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
        masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
            boundaries and resized to neural network output size.

        Note: Returned arrays might be zero padded if not enough target ROIs.
        """
        # Assertions
        asserts = [
            tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                    name="roi_assertion"),
        ]
        with tf.control_dependencies(asserts):
            proposals = tf.identity(proposals)

        # Remove zero padding
        proposals, _ = MaskRCNN.trim_zeros_graph(proposals, name="trim_proposals")
        gt_boxes, non_zeros = MaskRCNN.trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
        gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2, name="trim_gt_masks")

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
        non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
        crowd_boxes = tf.gather(gt_boxes, crowd_ix)
        gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
        gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
        gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

        # Compute overlaps matrix [proposals, gt_boxes]
        overlaps = MaskRCNN.overlaps_graph(proposals, gt_boxes)

        # Compute overlaps with crowd boxes [proposals, crowd_boxes]
        crowd_overlaps = MaskRCNN.overlaps_graph(proposals, crowd_boxes)
        crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)

        # Determine positive and negative ROIs
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
        positive_roi_bool = (roi_iou_max >= 0.5)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
        negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(config["TRAIN_ROIS_PER_IMAGE"] * config["ROI_POSITIVE_RATIO"])
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / config["ROI_POSITIVE_RATIO"]
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.cond(
            tf.greater(tf.shape(positive_overlaps)[1], 0),
            true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
            false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
        )
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

        # Compute bbox refinement for positive ROIs
        deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
        deltas /= config["BBOX_STD_DEV"]

        # Assign positive ROIs to GT masks
        # Permute masks to [N, height, width, 1]
        transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
        # Pick the right mask for each ROI
        roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

        # Compute mask targets
        boxes = positive_rois

        # if config["USE_MINI_MASK"]:
        #     # Transform ROI coordinates from normalized image space
        #     # to normalized mini-mask space.
        #     y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        #     gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        #     gt_h = gt_y2 - gt_y1
        #     gt_w = gt_x2 - gt_x1
        #     y1 = (y1 - gt_y1) / gt_h
        #     x1 = (x1 - gt_x1) / gt_w
        #     y2 = (y2 - gt_y1) / gt_h
        #     x2 = (x2 - gt_x1) / gt_w
        #     boxes = tf.concat([y1, x1, y2, x2], 1)

        box_ids = tf.range(0, tf.shape(roi_masks)[0])
        masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                        box_ids,
                                        config["MASK_SHAPE"])
        # Remove the extra dimension from masks.
        masks = tf.squeeze(masks, axis=3)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = tf.round(masks)

        # Append negative ROIs and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(config["TRAIN_ROIS_PER_IMAGE"] - tf.shape(rois)[0], 0)
        rois = tf.pad(rois, [(0, P), (0, 0)])
        roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
        deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
        masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

        return rois, roi_gt_class_ids, deltas, masks

    class BatchNorm(tf.keras.layers.BatchNormalization):
        """Extends the Keras BatchNormalization class to allow a central place
        to make changes if needed.

        Batch normalization has a negative effect on training if batches are small
        so this layer is often frozen (via setting in Config class) and functions
        as linear layer.
        """
        def call(self, inputs, training=None):
            """
            Note about training values:
                None: Train BN layers. This is the normal mode
                False: Freeze BN layers. Good when batch size is small
                True: (don't use). Set layer in training mode even when making inferences
            """
            return super(self.__class__, self).call(inputs, training=training) #self.__class__ ?

    def fpn_classifier_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True, fc_layers_size=1024):
        """Builds the computation graph of the feature pyramid network classifier
        and regressor heads.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
            coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                    [P2, P3, P4, P5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers
        fc_layers_size: Size of the 2 FC layers

        Returns:
            logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
            probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
            bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                        proposal boxes
        """
        # ROI Pooling
        # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
        x = MaskRCNN.PyramidROIAlign([pool_size, pool_size],
                            name="roi_align_classifier")([rois, image_meta] + feature_maps)
        # Two 1024 FC layers (implemented with Conv2D for consistency)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                            name="mrcnn_class_conv1")(x)
        x = tf.keras.layers.TimeDistributed(MaskRCNN.BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(fc_layers_size, (1, 1)),
                            name="mrcnn_class_conv2")(x)
        x = tf.keras.layers.TimeDistributed(MaskRCNN.BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
        x = tf.keras.layers.Activation('relu')(x)

        shared = tf.keras.layers.Lambda(lambda x: tf.squeeze(tf.squeeze(x, 3), 2),
                        name="pool_squeeze")(x)

        # Classifier head
        mrcnn_class_logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes),
                                                name='mrcnn_class_logits')(shared)
        mrcnn_probs = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("softmax"),
                                        name="mrcnn_class")(mrcnn_class_logits)

        # BBox head
        # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes * 4, activation='linear'),
                            name='mrcnn_bbox_fc')(shared)
        # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
        s = tf.keras.backend.int_shape(x)  # not working
        # print("check check check ", s)
        if s[1] == None:
            mrcnn_bbox = tf.keras.layers.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
        else:
            mrcnn_bbox = tf.keras.layers.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x) #change


        return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

    def build_fpn_mask_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True):
        """Builds the computation graph of the mask head of Feature Pyramid Network.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
            coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                    [P2, P3, P4, P5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers

        Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
        """
        # ROI Pooling
        # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
        x = MaskRCNN.PyramidROIAlign([pool_size, pool_size],
                            name="roi_align_mask")([rois, image_meta] + feature_maps)

        # Conv layers
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), padding="same"),
                            name="mrcnn_mask_conv1")(x)
        x = tf.keras.layers.TimeDistributed(MaskRCNN.BatchNorm(),
                            name='mrcnn_mask_bn1')(x, training=train_bn)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), padding="same"),
                            name="mrcnn_mask_conv2")(x)
        x = tf.keras.layers.TimeDistributed(MaskRCNN.BatchNorm(),
                            name='mrcnn_mask_bn2')(x, training=train_bn)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), padding="same"),
                            name="mrcnn_mask_conv3")(x)
        x = tf.keras.layers.TimeDistributed(MaskRCNN.BatchNorm(),
                            name='mrcnn_mask_bn3')(x, training=train_bn)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), padding="same"),
                            name="mrcnn_mask_conv4")(x)
        x = tf.keras.layers.TimeDistributed(MaskRCNN.BatchNorm(),
                            name='mrcnn_mask_bn4')(x, training=train_bn)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                            name="mrcnn_mask_deconv")(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                            name="mrcnn_mask")(x)
        return x

    class PyramidROIAlign(tf.keras.layers.Layer):
        """Implements ROI Pooling on multiple levels of the feature pyramid.

        Params:
        - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

        Inputs:
        - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                coordinates. Possibly padded with zeros if not enough
                boxes to fill the array.
        - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        - feature_maps: List of feature maps from different levels of the pyramid.
                        Each is [batch, height, width, channels]

        Output:
        Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
        The width and height are those specific in the pool_shape in the layer
        constructor.
        """

        def log2_graph(x):
            """Implementation of Log2. TF doesn't have a native implementation."""
            return tf.math.log(x) / tf.math.log(2.0)

        def __init__(self, pool_shape, **kwargs):
            super(MaskRCNN.PyramidROIAlign, self).__init__(**kwargs)
            self.pool_shape = tuple(pool_shape)

        def call(self, inputs):
            # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
            boxes = inputs[0]

            # Image meta
            # Holds details about the image. See compose_image_meta()
            image_meta = inputs[1]

            # Feature Maps. List of feature maps from different level of the
            # feature pyramid. Each is [batch, height, width, channels]
            feature_maps = inputs[2:]

            # Assign each ROI to a level in the pyramid based on the ROI area.
            y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
            h = y2 - y1
            w = x2 - x1
            # Use shape of first image. Images in a batch must have the same size.
            image_shape = MaskRCNN.parse_image_meta_graph(image_meta)['image_shape'][0]
            # Equation 1 in the Feature Pyramid Networks paper. Account for
            # the fact that our coordinates are normalized here.
            # e.g. a 224x224 ROI (in pixels) maps to P4
            image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
            roi_level = MaskRCNN.PyramidROIAlign.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
            roi_level = tf.minimum(5, tf.maximum(
                2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
            roi_level = tf.squeeze(roi_level, 2)

            # Loop through levels and apply ROI pooling to each. P2 to P5.
            pooled = []
            box_to_level = []
            for i, level in enumerate(range(2, 6)):
                ix = tf.where(tf.equal(roi_level, level))
                level_boxes = tf.gather_nd(boxes, ix)

                # Box indices for crop_and_resize.
                box_indices = tf.cast(ix[:, 0], tf.int32)

                # Keep track of which box is mapped to which level
                box_to_level.append(ix)

                # Stop gradient propogation to ROI proposals
                level_boxes = tf.stop_gradient(level_boxes)
                box_indices = tf.stop_gradient(box_indices)

                # Crop and Resize
                # From Mask R-CNN paper: "We sample four regular locations, so
                # that we can evaluate either max or average pooling. In fact,
                # interpolating only a single value at each bin center (without
                # pooling) is nearly as effective."
                #
                # Here we use the simplified approach of a single value per bin,
                # which is how it's done in tf.crop_and_resize()
                # Result: [batch * num_boxes, pool_height, pool_width, channels]
                pooled.append(tf.image.crop_and_resize(
                    feature_maps[i], level_boxes, box_indices, self.pool_shape,
                    method="bilinear"))

            # Pack pooled features into one tensor
            pooled = tf.concat(pooled, axis=0)

            # Pack box_to_level mapping into one array and add another
            # column representing the order of pooled boxes
            box_to_level = tf.concat(box_to_level, axis=0)
            box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
            box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                    axis=1)

            # Rearrange pooled features to match the order of the original boxes
            # Sort box_to_level by batch then box index
            # TF doesn't have a way to sort by two columns, so merge them and sort.
            sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
            ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
                box_to_level)[0]).indices[::-1]
            ix = tf.gather(box_to_level[:, 2], ix)
            pooled = tf.gather(pooled, ix)

            # Re-add the batch dimension
            shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
            pooled = tf.reshape(pooled, shape)
            return pooled

        def compute_output_shape(self, input_shape):
            return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )

    class AnchorsLayer(tf.keras.layers.Layer):
        def __init__(self, anchors, name="anchors", **kwargs):
            super(MaskRCNN.AnchorsLayer, self).__init__(name=name, **kwargs)
            self.anchors = tf.Variable(anchors)

        def call(self, dummy):
            return self.anchors

        def get_config(self):
            config = super(MaskRCNN.AnchorsLayer, self).get_config()
            return config

    class ROIAlignLayer(tf.keras.layers.Layer):

        @staticmethod
        def log2_graph(x):
            """Implementation of Log2. TF doesn't have a native implementation."""
            return tf.log(x) / tf.log(2.0)

        def __init__(self, pool_shape, **kwargs):
            super(MaskRCNN.ROIAlignLayer,self).__init__( **kwargs)
            self.pool_shape = tuple(pool_shape)

        def call(self,inputs):
            # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
            boxes = inputs[0]

            # Image meta
            # Holds details about the image. See compose_image_meta()
            image_meta = inputs[1]

            # Feature Maps
            # List of feature maps from different level of the feature pyramid
            # Each is [batch, height, width, channels]
            feature_maps = inputs[2:]

            # Assign each ROI to a level in the pyramid based on the ROI area.
            y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
            h = y2 - y1
            w = x2 - x1

            # Use shape of first image. Images in a batch must have the same size.
            image_shape = MaskRCNN.parse_image_meta_graph(image_meta)['image_shape'][0]

            # Equation 1 in the Feature Pyramid Networks paper. Account for
            # the fact that our coordinates are normalized here.
            # e.g. a 224x224 ROI (in pixels) maps to P4
            image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
            roi_level = MaskRCNN.ROIAlignLayer.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
            roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
            roi_level = tf.squeeze(roi_level, 2)

            # Loop through levels and apply ROI pooling to each. P2 to P5.
            pooled = []
            box_to_level = []
            for i, level in enumerate(range(2, 6)):
                ix = tf.where(tf.equal(roi_level, level))
                level_boxes = tf.gather_nd(boxes, ix)

                # Box indices for crop_and_resize.
                box_indices = tf.cast(ix[:, 0], tf.int32)

                # Keep track of which box is mapped to which level
                box_to_level.append(ix)

                # Stop gradient propogation to ROI proposals
                level_boxes = tf.stop_gradient(level_boxes)
                box_indices = tf.stop_gradient(box_indices)

                # Crop and Resize
                # From Mask R-CNN paper: "We sample four regular locations, so
                # that we can evaluate either max or average pooling. In fact,
                # interpolating only a single value at each bin center (without
                # pooling) is nearly as effective."
                #
                # Here we use the simplified approach of a single value per bin,
                # which is how it's done in tf.crop_and_resize()
                # Result: [batch * num_boxes, pool_height, pool_width, channels]
                pooled.append(tf.image.crop_and_resize(
                    feature_maps[i], level_boxes, box_indices, self.pool_shape,
                    method="bilinear"))

            # Pack pooled features into one tensor
            pooled = tf.concat(pooled, axis=0)

            # Pack box_to_level mapping into one array and add another
            # column representing the order of pooled boxes
            box_to_level = tf.concat(box_to_level, axis=0)
            box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
            box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                    axis=1)

            # Rearrange pooled features to match the order of the original boxes
            # Sort box_to_level by batch then box index
            # TF doesn't have a way to sort by two columns, so merge them and sort.
            sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
            ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
                box_to_level)[0]).indices[::-1]
            ix = tf.gather(box_to_level[:, 2], ix)
            pooled = tf.gather(pooled, ix)

            # Re-add the batch dimension
            shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
            pooled = tf.reshape(pooled, shape)
            return pooled

    class ProposalLayer(tf.keras.layers.Layer):
        
        @staticmethod
        def apply_box_deltas_graph(boxes, deltas):
            """Applies the given deltas to the given boxes.
            boxes: [N, (y1, x1, y2, x2)] boxes to update
            deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
            """
            # Convert to y, x, h, w
            height = boxes[:, 2] - boxes[:, 0]
            width = boxes[:, 3] - boxes[:, 1]
            center_y = boxes[:, 0] + 0.5 * height
            center_x = boxes[:, 1] + 0.5 * width
            # Apply deltas
            center_y += deltas[:, 0] * height
            center_x += deltas[:, 1] * width
            height *= tf.exp(deltas[:, 2])
            width *= tf.exp(deltas[:, 3])
            # Convert back to y1, x1, y2, x2
            y1 = center_y - 0.5 * height
            x1 = center_x - 0.5 * width
            y2 = y1 + height
            x2 = x1 + width
            result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out") # pattern: y1, x1, y2, x2
            return result

        @staticmethod
        def clip_boxes_graph(boxes, window):
            """
            boxes: [N, (y1, x1, y2, x2)]
            window: [4] in the form y1, x1, y2, x2
            """
            # Split
            wy1, wx1, wy2, wx2 = tf.split(window, 4)
            y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
            # Clip
            y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
            x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
            y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
            x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
            clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
            clipped.set_shape((clipped.shape[0], 4))
            return clipped

        """Receives anchor scores and selects a subset to pass as proposals
        to the second stage. Filtering is done based on anchor scores and
        non-max suppression to remove overlaps. It also applies bounding
        box refinement deltas to anchors.

        Inputs:
            rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
            rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
            anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

        Returns:
            Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
        """

        def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
            super(MaskRCNN.ProposalLayer, self).__init__(**kwargs)
            self.config = config
            self.proposal_count = proposal_count
            self.nms_threshold = nms_threshold

        def call(self, inputs):
            # [rpn_class, rpn_bbox, anchor_layer]
            """
            inputs[0] = rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
            inputs[1] = rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be applied to anchors.
            """
            # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
            scores = inputs[0][:, :, 1]
            print("[info] scores.shape ",scores.shape,np.array(inputs).shape)
            """
            >>> Why use foreground confidence only?
            >>> foreground means objects that appear in front of the background
            >>> so we need to try to find whether there are object of interest (OFI) that we may concern...
            """
            # Box deltas [batch, num_rois, 4]
            deltas = inputs[1]
            deltas = deltas * np.reshape(self.config["RPN_BBOX_STD_DEV"], [1, 1, 4])
            """
            >>> Why do deltas * np.reshape(self.config["RPN_BBOX_STD_DEV"], [1, 1, 4])?
            When preparing the training targets for the regressor, subtract the mean and divide by the standard deviation. 
            This will give the regressor normalized training targets. That also means that the regressor is going to predict normalized values 
            rather than the actual values we would use to shift and resize the anchors.
            After training is complete, when we want to use the output from the regressor, we multiply by the standard deviation 
            and add the mean to reverse the transformation in 1. This converts the predicted normalized values into real values we can 
            use to shift and resize the anchor boxes.
            """
            # Anchors
            anchors = inputs[2]

            # Improve performance by trimming to top anchors by score
            # and doing the rest on the smaller subset.
            print("[info] anchors shape",anchors.shape)
            print("[info] PRE_NMS_LIMIT ",self.config["PRE_NMS_LIMIT"])
            """
            >>> Why do we need pre NMS limit?
            """
            pre_nms_limit = tf.minimum(self.config["PRE_NMS_LIMIT"], tf.shape(anchors)[1])
            # pre nms limit: ROIs kept after tf.nn.top_k and before non-maximum suppression
            # top_k: Finds values and indices of the k largest entries for the last dimension.
            # input, k=1, sorted=True, name=None
            ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
            scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.config["IMAGES_PER_GPU"])
            deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.config["IMAGES_PER_GPU"])
            pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), self.config["IMAGES_PER_GPU"], names=["pre_nms_anchors"])

            # Apply deltas to anchors to get refined anchors.
            # [batch, N, (y1, x1, y2, x2)]
            boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                    lambda x, y: MaskRCNN.ProposalLayer.apply_box_deltas_graph(x, y),
                                    self.config["IMAGES_PER_GPU"],
                                    names=["refined_anchors"])

            # Clip to image boundaries. Since we're in normalized coordinates,
            # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
            window = np.array([0, 0, 1, 1], dtype=np.float32)
            boxes = utils.batch_slice(boxes,
                                    lambda x: MaskRCNN.ProposalLayer.clip_boxes_graph(x, window),
                                    self.config["IMAGES_PER_GPU"],
                                    names=["refined_anchors_clipped"])

            # Filter out small boxes
            # According to Xinlei Chen's paper, this reduces detection accuracy
            # for small objects, so we're skipping it.

            # Non-max suppression
            def nms(boxes, scores):
                indices = tf.image.non_max_suppression(
                    boxes, scores, self.proposal_count,
                    self.nms_threshold, name="rpn_non_max_suppression")
                proposals = tf.gather(boxes, indices)
                # Pad if needed
                padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
                proposals = tf.pad(proposals, [(0, padding), (0, 0)])
                return proposals

            proposals = utils.batch_slice([boxes, scores], nms, self.config["IMAGES_PER_GPU"])

            return proposals

        def compute_output_shape(self, input_shape):
            return (None, self.proposal_count, 4)

    class DetectionTargetLayer(tf.keras.layers.Layer):
        """Subsamples proposals and generates target box refinement, class_ids,
        and masks for each.

        Inputs:
        proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
                be zero padded if there are not enough proposals.
        gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
        gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
                coordinates.
        gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

        Returns: Target ROIs and corresponding class IDs, bounding box shifts,
        and masks.
        rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
            coordinates
        target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
        target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                    Masks cropped to bbox boundaries and resized to neural
                    network output size.

        Note: Returned arrays might be zero padded if not enough target ROIs.
        """

        def __init__(self, config, **kwargs):
            super(MaskRCNN.DetectionTargetLayer, self).__init__(**kwargs)
            self.config = config

        def call(self, inputs):
            proposals = inputs[0]
            gt_class_ids = inputs[1]
            gt_boxes = inputs[2]
            gt_masks = inputs[3]

            print("proposals shape",proposals.shape)
            print("gt_class_ids shape",gt_class_ids.shape)
            print("gt_boxes shape",gt_boxes.shape)
            print("gt_masks shape",gt_masks.shape)

            # Slice the batch and run a graph for each slice
            # TODO: Rename target_bbox to target_deltas for clarity
            names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
            outputs = utils.batch_slice(
                [proposals, gt_class_ids, gt_boxes, gt_masks],
                lambda w, x, y, z: MaskRCNN.detection_targets_graph(w, x, y, z, self.config),
                self.config["IMAGES_PER_GPU"], names=names)
            return outputs

        def compute_output_shape(self, input_shape):
            return [
                (None, self.config["TRAIN_ROIS_PER_IMAGE"], 4),  # rois
                (None, self.config["TRAIN_ROIS_PER_IMAGE"]),  # class_ids
                (None, self.config["TRAIN_ROIS_PER_IMAGE"], 4),  # deltas
                (None, self.config["TRAIN_ROIS_PER_IMAGE"],
                self.config["MASK_SHAPE"][0],
                self.config["MASK_SHAPE"][1])  # masks
            ]

        def compute_mask(self, inputs, mask=None):
            return [None, None, None, None]

    ############################################################
    #  Loss Functions
    ############################################################
    @staticmethod
    def batch_pack_graph(x, counts, num_rows):
        """Picks different number of values from each row
        in x depending on the values in counts.
        """
        outputs = []
        for i in range(num_rows):
            outputs.append(x[i, :counts[i]])
        return tf.concat(outputs, axis=0)
    
    @staticmethod
    def smooth_l1_loss(y_true, y_pred):
        """Implements Smooth-L1 loss.
        y_true and y_pred are typically: [N, 4], but could be any shape.
        """
        diff = tf.abs(y_true - y_pred)
        less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
        loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
        return loss

    def rpn_class_loss_graph(rpn_match, rpn_class_logits):
        """RPN anchor classifier loss.

        rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                -1=negative, 0=neutral anchor.
        rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
        """
        # Squeeze last dim to simplify
        rpn_match = tf.squeeze(rpn_match, -1)
        # Get anchor classes. Convert the -1/+1 match to 0/1 values.
        anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
        # Positive and Negative anchors contribute to the loss,
        # but neutral anchors (match value = 0) don't.
        indices = tf.where(tf.not_equal(rpn_match, 0))
        # Pick rows that contribute to the loss and filter out the rest.
        rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
        anchor_class = tf.gather_nd(anchor_class, indices)
        # Cross entropy loss # not working
        loss = tf.keras.metrics.sparse_categorical_crossentropy(y_true=anchor_class,
                                                y_pred=rpn_class_logits,
                                                from_logits=True)
        loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.keras.backend.mean(loss), tf.constant(0.0))
        return loss

    def rpn_bbox_loss_graph(images_per_gpu, target_bbox, rpn_match, rpn_bbox):
        """Return the RPN bounding box loss graph.

        config: the model config object.
        target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
            Uses 0 padding to fill in unsed bbox deltas.
        rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                -1=negative, 0=neutral anchor.
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        """
        # Positive anchors contribute to the loss, but negative and
        # neutral anchors (match value of 0 or -1) don't.
        rpn_match = tf.squeeze(rpn_match, -1)
        indices = tf.where(tf.equal(rpn_match, 1))

        # Pick bbox deltas that contribute to the loss
        rpn_bbox = tf.gather_nd(rpn_bbox, indices)

        # Trim target bounding box deltas to the same length as rpn_bbox.
        batch_counts = tf.keras.backend.sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
        target_bbox = MaskRCNN.batch_pack_graph(target_bbox, batch_counts,
                                    images_per_gpu)

        loss = MaskRCNN.smooth_l1_loss(target_bbox, rpn_bbox)

        loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.keras.backend.mean(loss), tf.constant(0.0))
        return loss

    def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
        """Loss for the classifier head of Mask RCNN.

        target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
            padding to fill in the array.
        pred_class_logits: [batch, num_rois, num_classes]
        active_class_ids: [batch, num_classes]. Has a value of 1 for
            classes that are in the dataset of the image, and 0
            for classes that are not in the dataset.
        """
        # During model building, Keras calls this function with
        # target_class_ids of type float32. Unclear why. Cast it
        # to int to get around it.
        target_class_ids = tf.cast(target_class_ids, 'int64')

        # Find predictions of classes that are not in the dataset.
        pred_class_ids = tf.argmax(pred_class_logits, axis=2)
        # TODO: Update this line to work with batch > 1. Right now it assumes all
        #       images in a batch have the same active_class_ids
        pred_active = tf.gather(active_class_ids[0], pred_class_ids)

        # Loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_class_ids, logits=pred_class_logits)

        # Erase losses of predictions of classes that are not in the active
        # classes of the image.
        loss = loss * pred_active

        # Computer loss mean. Use only predictions that contribute
        # to the loss to get a correct mean.
        loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
        return loss

    def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
        """Loss for Mask R-CNN bounding box refinement.

        target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
        target_class_ids: [batch, num_rois]. Integer class IDs.
        pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        """
        # Reshape to merge batch and roi dimensions for simplicity.
        target_class_ids = tf.reshape(target_class_ids, (-1,))
        target_bbox = tf.reshape(target_bbox, (-1, 4))
        pred_bbox = tf.reshape(pred_bbox, (-1, tf.keras.backend.int_shape(pred_bbox)[2], 4))

        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indices.
        positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = tf.cast(
            tf.gather(target_class_ids, positive_roi_ix), tf.int64)
        indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = tf.gather(target_bbox, positive_roi_ix)
        pred_bbox = tf.gather_nd(pred_bbox, indices)

        # Smooth-L1 Loss
        loss = tf.keras.backend.switch(tf.size(target_bbox) > 0,
                        MaskRCNN.smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                        tf.constant(0.0))
        loss = tf.keras.backend.mean(loss)
        return loss

    def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
        """Mask binary cross-entropy loss for the masks head.

        target_masks: [batch, num_rois, height, width].
            A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                    with values from 0 to 1.
        """
        # Reshape for simplicity. Merge first two dimensions into one.
        target_class_ids = tf.reshape(target_class_ids, (-1,))
        mask_shape = tf.shape(target_masks)
        target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
        pred_shape = tf.shape(pred_masks)
        pred_masks = tf.reshape(pred_masks,
                            (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
        # Permute predicted masks to [N, num_classes, height, width]
        pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = tf.where(target_class_ids > 0)[:, 0]
        positive_class_ids = tf.cast(
            tf.gather(target_class_ids, positive_ix), tf.int64)
        indices = tf.stack([positive_ix, positive_class_ids], axis=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = tf.gather(target_masks, positive_ix)
        y_pred = tf.gather_nd(pred_masks, indices)

        # Compute binary cross entropy. If no positive ROIs, then return 0.
        # shape: [batch, roi, num_classes]
        loss = tf.keras.backend.switch(tf.size(y_true) > 0,
                        tf.keras.backend.binary_crossentropy(target=y_true, output=y_pred),
                        tf.constant(0.0))
        loss = tf.keras.backend.mean(loss)
        return loss

    ### MODEL CONFIG
    CHANNEL_SIZE = 256
    LAYER_NUM = 4
    RPN_MATCH_SIZE = 1
    RPN_BOX_SIZE = 4
    BASE_MODEL_STRIDES = [4, 2, 2, 2]
    P6_STRIDE = 2
    ANCHOR_AREAS = [32**2,64**2,128**2,256**2,512**2]
    ASPECT_RATIOS = [(1,2),(1,1),(2,1)]
    ANCHOR_STRIDE = 1
    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    # Bounding box refinement standard deviation for RPN and final detections.
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 6000
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200
    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]
    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33
    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small
     # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    


    def __init__(self,basemodel):
        
        self.basemodel = basemodel
        self.resnet_layer1 = basemodel.layer1
        self.resnet_layer2 = basemodel.layer2
        self.resnet_layer3 = basemodel.layer3
        self.resnet_layer4 = basemodel.layer4

        self.conv1x1_layers = []
        for _ in range(MaskRCNN.LAYER_NUM):
            self.conv1x1_layers.append(tf.keras.layers.Conv2D(filters=MaskRCNN.CHANNEL_SIZE, kernel_size=1, activation=None, padding='SAME'))

        self.conv3x3_layers = []
        for _ in range(MaskRCNN.LAYER_NUM):
            self.conv3x3_layers.append(tf.keras.layers.Conv2D(filters=MaskRCNN.CHANNEL_SIZE, kernel_size=3, activation=None, padding='SAME'))

        self.upsample_layers = []
        for _ in range(MaskRCNN.LAYER_NUM-1):
            self.upsample_layers.append(tf.keras.layers.UpSampling2D(size=(2,2)))

        self.maxpooling = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(2,2), padding='SAME')


    def forward(self,num_classes,sample_input,batch_size=1):
        no_of_gpu = 1
        MaskRCNN.IMAGES_PER_GPU = batch_size // no_of_gpu # batch size can be no. of gpu * images per gpu; here assume no of gpu = 1
        assert MaskRCNN.IMAGES_PER_GPU > 0
        ### step1: prepare inputs
        input_image = tf.keras.Input(shape=(None, None, sample_input.shape[3])) # only provides C dim, why doesnt provide h w here?
        input_image_meta = tf.keras.Input(shape=[MaskRCNN.get_compose_image_meta_len(num_classes)],name="input_image_meta")
           
        #no input_image_meta
        input_rpn_match = tf.keras.Input(shape=[None, MaskRCNN.RPN_MATCH_SIZE], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = tf.keras.Input(shape=[None, MaskRCNN.RPN_BOX_SIZE], name="input_rpn_bbox", dtype=sample_input.dtype)
        # 1. GT Class IDs (zero padded)
        input_gt_class_ids = tf.keras.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
        # 2. GT Boxes in pixels (zero padded)
        input_gt_boxes = tf.keras.Input(shape=[None, MaskRCNN.RPN_BOX_SIZE], name="input_gt_boxes", dtype=sample_input.dtype)
        gt_boxes = tf.keras.layers.Lambda(lambda x: MaskRCNN.norm_boxes_graph(x))([input_gt_boxes,input_image])
        # USE_MINI_MASK is FALSE
        input_gt_masks = tf.keras.Input(shape=[sample_input.shape[1],sample_input.shape[2], None],name="input_gt_masks", dtype=bool)
        print(">>> input_image",input_image,sample_input.shape)
        print(">>> input_rpn_match",input_rpn_match)
        print(">>> input_rpn_bbox",input_rpn_bbox)
        print(">>> input_gt_class_ids",input_gt_class_ids)
        print(">>> input_gt_boxes",input_gt_boxes)


        ### step2: prepare layers
        c2 = self.resnet_layer1(self.basemodel,input_image)
        c3 = self.resnet_layer2(self.basemodel,c2)
        c4 = self.resnet_layer3(self.basemodel,c3)
        c5 = self.resnet_layer4(self.basemodel,c4)
        # print('sample',sample_input.shape,c2.shape,c3.shape,c4.shape,c5.shape)

        M5 = self.conv1x1_layers[0](c5)
        M4 = self.upsample_layers[0](M5) + self.conv1x1_layers[1](c4)
        M3 = self.upsample_layers[1](M4) + self.conv1x1_layers[2](c3)
        M2 = self.upsample_layers[2](M3) + self.conv1x1_layers[3](c2)

        P2 = self.conv3x3_layers[3](M2)
        P3 = self.conv3x3_layers[2](M3)
        P4 = self.conv3x3_layers[1](M4)
        P5 = self.conv3x3_layers[0](M5)
        # P6 is used for the 5th anchor scale in RPN. Generated by subsampling from P5 with stride of 2.
        P6 = self.maxpooling(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]
        
        ### step3: Anchors shape anchors (261888, 4)
        # print("an shapes",P2.shape[1:3],P3.shape[1:3],P4.shape[1:3],P5.shape[1:3],P6.shape[1:3])
        feature_shapes = [P2.shape[1:3],P3.shape[1:3],P4.shape[1:3],P5.shape[1:3],P6.shape[1:3]]
        anchors_arr = self.gen_anchors(sample_input.shape[1:3],feature_shapes,MaskRCNN.ANCHOR_AREAS,MaskRCNN.ASPECT_RATIOS,MaskRCNN.ANCHOR_STRIDE)
        print("batch_size",batch_size)
        anchors = np.broadcast_to(anchors_arr, (batch_size,) + anchors_arr.shape)  # insert batch_size in front of the anchors_arr.shape
        # anchors_arr.shape = [N, (y1, x1, y2, x2)] in normalized coordinates
        anchor_layer = MaskRCNN.AnchorsLayer(anchors, name="anchors")(input_image)
        print("anchor_layer",anchor_layer.shape)

        ### RPN
        rpn_class_logits, rpn_class, rpn_bbox = self.build_rpn_model(MaskRCNN.ANCHOR_STRIDE,len(MaskRCNN.ASPECT_RATIOS),MaskRCNN.CHANNEL_SIZE,rpn_feature_maps)
        
        ## Gen proposals
        rpn_config = {"RPN_BBOX_STD_DEV":MaskRCNN.RPN_BBOX_STD_DEV,"PRE_NMS_LIMIT":MaskRCNN.PRE_NMS_LIMIT,"IMAGES_PER_GPU":MaskRCNN.IMAGES_PER_GPU}
        rpn_rois = MaskRCNN.ProposalLayer(
            proposal_count=MaskRCNN.POST_NMS_ROIS_TRAINING,
            nms_threshold=MaskRCNN.RPN_NMS_THRESHOLD,
            name="ROI",
            config=rpn_config)([rpn_class, rpn_bbox, anchor_layer])
        
        ## Class ID mask to mark class IDs supported by the dataset the image came from.
        active_class_ids = tf.keras.layers.Lambda(lambda x: MaskRCNN.parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)
        print(active_class_ids)
        
        target_rois = rpn_rois # use rpn rois from rpn as target rois

        ## Gen detection targets
        detect_config = {
            "TRAIN_ROIS_PER_IMAGE":MaskRCNN.TRAIN_ROIS_PER_IMAGE,
            "MASK_SHAPE":MaskRCNN.MASK_SHAPE,
            "IMAGES_PER_GPU":MaskRCNN.IMAGES_PER_GPU,
            "ROI_POSITIVE_RATIO":MaskRCNN.ROI_POSITIVE_RATIO,
            "BBOX_STD_DEV":MaskRCNN.BBOX_STD_DEV,
            "MASK_SHAPE":MaskRCNN.MASK_SHAPE
        }
        rois, target_class_ids, target_bbox, target_mask =\
            MaskRCNN.DetectionTargetLayer(detect_config, name="proposal_targets")([target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

        # return

        ### Network HEADS
        fpn_config = {
            "POOL_SIZE":MaskRCNN.POOL_SIZE,
            "NUM_CLASSES":num_classes,
            "TRAIN_BN":MaskRCNN.TRAIN_BN,
            "FPN_CLASSIF_FC_LAYERS_SIZE":MaskRCNN.FPN_CLASSIF_FC_LAYERS_SIZE,
            "MASK_POOL_SIZE":MaskRCNN.MASK_POOL_SIZE
        }
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
            MaskRCNN.fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                    fpn_config["POOL_SIZE"], fpn_config["NUM_CLASSES"],
                                    train_bn=fpn_config["TRAIN_BN"],
                                    fc_layers_size=fpn_config["FPN_CLASSIF_FC_LAYERS_SIZE"])

        mrcnn_mask = MaskRCNN.build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                            input_image_meta,
                                            fpn_config["MASK_POOL_SIZE"],
                                            fpn_config["NUM_CLASSES"],
                                            train_bn=fpn_config["TRAIN_BN"])
        # return

        ### define Losses
        output_rois = tf.keras.layers.Lambda(lambda x: x * 1, name="output_rois")(rois)
        rpn_class_loss = tf.keras.layers.Lambda(lambda x: MaskRCNN.rpn_class_loss_graph(*x), name="rpn_class_loss")(
            [input_rpn_match, rpn_class_logits])
        rpn_bbox_loss = tf.keras.layers.Lambda(lambda x: MaskRCNN.rpn_bbox_loss_graph(MaskRCNN.IMAGES_PER_GPU, *x), name="rpn_bbox_loss")(
            [input_rpn_bbox, input_rpn_match, rpn_bbox])
        class_loss = tf.keras.layers.Lambda(lambda x: MaskRCNN.mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
            [target_class_ids, mrcnn_class_logits, active_class_ids])
        bbox_loss = tf.keras.layers.Lambda(lambda x: MaskRCNN.mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
            [target_bbox, target_class_ids, mrcnn_bbox])
        mask_loss = tf.keras.layers.Lambda(lambda x: MaskRCNN.mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
            [target_mask, target_class_ids, mrcnn_mask])

        ### define Model

        print("input_image",input_image)
        print("input_image_meta",input_image_meta)
        print("input_rpn_match",input_rpn_match)
        print("input_rpn_bbox",input_rpn_bbox)
        print("input_gt_class_ids",input_gt_class_ids)
        print("input_gt_boxes",input_gt_boxes)
        print("input_gt_masks",input_gt_masks)

        print("rpn_class_logits",rpn_class_logits)
        print("rpn_class",rpn_class)
        print("rpn_bbox",rpn_bbox)
        print("mrcnn_class_logits",mrcnn_class_logits)
        print("mrcnn_class",mrcnn_class)
        print("mrcnn_bbox",mrcnn_bbox)
        print("mrcnn_mask",mrcnn_mask)
        print("rpn_rois",rpn_rois)
        print("output_rois",output_rois)
        print("rpn_bbox_loss",rpn_bbox_loss)
        print("class_loss",class_loss)
        print("bbox_loss",bbox_loss)
        print("mask_loss",mask_loss)

        inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]

        outputs = [rpn_class_logits, rpn_class, rpn_bbox, 
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, 
                rpn_rois, output_rois,
                rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]

        model = None
        model = tf.keras.Model(inputs, outputs, name='mask_rcnn')
        # import keras
        # import keras.models as KM
        # model = KM.Model(inputs, outputs, name='mask_rcnn')


        #TODO: add multi GPU support

        return model

basemodel = ResNetStruct().construct(50,strides=MaskRCNN.BASE_MODEL_STRIDES)
mymodel = MaskRCNN(basemodel)

if __name__ == "__main__":
    sample_input = tf.keras.Input(shape=(224,224,3),dtype=tf.float32) #input_shape = [224,224,3] #H,W,C
    out = mymodel.forward(10,sample_input,batch_size=2)
    # print(out.summary())