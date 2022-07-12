import numpy as np
import tensorflow as tf
try:
    import Image
except ImportError:
    from PIL import Image, ImageDraw

def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True, preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    # if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
    #     # New in 0.14: anti_aliasing. Default it to False for backward
    #     # compatibility with skimage 0.13.
    #     return skimage.transform.resize(
    #         image, output_shape,
    #         order=order, mode=mode, cval=cval, clip=clip,
    #         preserve_range=preserve_range, anti_aliasing=anti_aliasing,
    #         anti_aliasing_sigma=anti_aliasing_sigma)
    # else:
    import skimage.transform
    return skimage.transform.resize(
        image, output_shape,
        order=order, mode=mode, cval=cval, clip=clip,
        preserve_range=preserve_range)

def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """

    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)


    full_mask[y1:y2, x1:x2] = mask
    return full_mask

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

def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)

def unmold_detections(detections, mrcnn_mask, original_image_shape, image_shape, window):
    """Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.

    detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
    mrcnn_mask: [N, height, width, num_classes]
    original_image_shape: [H, W, C] Original image shape before resizing
    image_shape: [H, W, C] Shape of the image after resizing and padding
    window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
            image is excluding the padding.

    Returns:
    boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
    class_ids: [N] Integer class IDs for each bounding box
    scores: [N] Float probability scores of the class_id
    masks: [height, width, num_instances] Instance masks
    """
    # print("detections",detections.shape,detections)
    # print("mrcnn_mask",mrcnn_mask.shape,mrcnn_mask)
    #TODO: add back masks
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
    
    if N==0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    idx = detections[:N,6]
    masks = np.take(mrcnn_mask,np.array(idx,dtype=np.uint32),axis=0)
    # masks =  np.take(masks,class_ids,axis=-1)
    masks = masks[np.arange(N), :, :, class_ids]

    # Translate normalized coordinates in the resized image to pixel
    # coordinates in the original image before resizing
    window = norm_boxes(window, image_shape[:2])
    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    # Convert boxes to normalized coordinates on the window
    boxes = np.divide(boxes - shift, scale)
    # print("norm boxes", boxes.shape, boxes)
    # Convert boxes to pixel coordinates on the original image
    boxes = denorm_boxes(boxes, original_image_shape[:2])
    # print("boxes denorm", boxes.shape, boxes)

    # Filter out detections with zero area. Happens in early training when
    # network weights are still random

    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]

    # Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(N):
        # Convert neural network mask to full size mask
        full_mask = unmold_mask(masks[i], boxes[i], original_image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1)\
        if full_masks else np.empty(original_image_shape[:2] + (0,))

    return boxes, class_ids, scores, full_masks

def parse_image_meta_graph(meta):
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

def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = float(shape[0]), float(shape[1])
    scale = np.array([h, w, h, w]) - 1.0
    shift = np.array([0., 0., 1., 1.])
    
    return np.divide(boxes - shift, scale)

def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        # if not isinstance(output_slice, (tuple, list)):
        #     output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices

    # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)]

    # outputs = list(zip(*outputs))

    # if names is None:
    #     names = [None] * len(outputs)

    # result = [np.stack(o, axis=0, name=n) for o, n in zip(outputs, names)]
    # if len(result) == 1:
    #     result = result[0]

    return np.array(outputs)

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
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = np.stack([y1, x1, y2, x2], axis=1)
    return result

def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = np.split(window, 4)
    y1, x1, y2, x2 = np.split(boxes, 4, axis=1)
    # Clip
    y1 = np.maximum(np.minimum(y1, wy2), wy1)
    x1 = np.maximum(np.minimum(x1, wx2), wx1)
    y2 = np.maximum(np.minimum(y2, wy2), wy1)
    x2 = np.maximum(np.minimum(x2, wx2), wx1)
    clipped = np.concatenate([y1, x1, y2, x2], axis=1) #, name="clipped_boxes")
    # clipped.set_shape((clipped.shape[0], 4))
    return clipped

def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI

    class_ids = np.argmax(probs, axis=1).reshape(-1,1)
    # Class probability of the top class of each ROI
    # indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1) #change
    # indices = np.stack([np.range(np.shape(probs)[0]), class_ids], axis = 1)
    indices = class_ids
    class_scores = [] #np.gather_nd(probs, indices)
    for i in range(len(probs)):
        class_scores.append(np.take(probs[i],indices[i],axis=0)[0])
    class_scores = np.array(class_scores).reshape(-1,1)

    # Class-specific bounding box deltas
    deltas_specific = [] #np.gather_nd(deltas, indices)
    for i in range(len(probs)):
        deltas_specific.append(np.take(deltas[i],indices[i],axis=0)[0])
    deltas_specific = np.array(deltas_specific).reshape(-1,4)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    
    refined_rois = apply_box_deltas_graph(rois, deltas_specific * config['BBOX_STD_DEV'])
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area
    exclude_ix = np.where(
        (refined_rois[:, 2] - refined_rois[:, 0]) * (refined_rois[:, 3] - refined_rois[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        refined_rois = np.delete(refined_rois, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        class_scores = np.delete(class_scores, exclude_ix, axis=0)
        # masks = np.delete(masks, exclude_ix, axis=0)
        # N = class_ids.shape[0]

    # Filter out background boxes
    keep = np.where(class_ids > 0)[0] #[:, 0] #TEMP: get also class id == 0
    keep = set(keep.tolist())
    # Filter out low confidence boxes
    if config['DETECTION_MIN_CONFIDENCE']:
        conf_keep = np.where(class_scores >= config['DETECTION_MIN_CONFIDENCE'])[0] #[:, 0]

        conf_keep = set(conf_keep.tolist())
        keep = keep.intersection(conf_keep)
        # keep = np.sets.intersection(np.expand_dims(keep, 0), np.expand_dims(conf_keep, 0))
        # keep = np.sparse.to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables

    keep = np.array(list(keep)) # convert back to np array

    pre_nms_class_ids = np.take(class_ids, keep, axis=0).flatten()
    pre_nms_scores = np.take(class_scores, keep, axis=0).flatten()
    pre_nms_rois = np.take(refined_rois,   keep, axis=0)

    # print('pre_nms_class_ids',pre_nms_class_ids.shape)
    # print('pre_nms_scores',pre_nms_scores.shape)
    # print('pre_nms_rois',pre_nms_rois.shape)

    unique_pre_nms_class_ids = np.unique(pre_nms_class_ids)

    def nms_keep_map(class_id,pre_nms_class_ids,pre_nms_rois,pre_nms_scores):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = []
        for cid in class_id:
            ixs.append(np.where(np.equal(pre_nms_class_ids, cid))[0]) #[:, 0]
        # ixs = np.array(ixs)

        all_class_idx = set()

        for i in range(len(ixs)):
            # Apply NMS
            
            boxes = np.take(pre_nms_rois, ixs[i], axis=0)
            scores = np.take(pre_nms_scores, ixs[i])

            class_keep = tf.image.non_max_suppression(
                    boxes,
                    scores,
                    max_output_size=config['DETECTION_MAX_INSTANCES'],
                    iou_threshold=config['DETECTION_NMS_THRESHOLD'])

            class_keep = np.array(class_keep)

            # Map indices
            class_keep_idx = np.take(ixs[i], class_keep)

            # Pad with -1 so returned tensors have the same shape
            # gap = config['DETECTION_MAX_INSTANCES'] - np.shape(class_keep)[0]
            # class_keep = np.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)


            # Set shape so map_fn() can infer result shape
            # class_keep.set_shape([config['DETECTION_MAX_INSTANCES']])
        
            class_keep_idx = np.array(class_keep_idx,dtype=np.int64)

            all_class_idx = all_class_idx.union(set(class_keep_idx.tolist()))

        return np.array(list(all_class_idx))

    # 2. Map over class IDs
    # nms_keep = np.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=np.int64)
    nms_keep = nms_keep_map(unique_pre_nms_class_ids,pre_nms_class_ids,pre_nms_rois,pre_nms_scores)
    # 3. Merge results into one list, and remove -1 padding
    # nms_keep = np.reshape(nms_keep, [-1])
    # nms_keep = np.take(nms_keep, np.where(nms_keep > -1)[:, 0])
    # # 4. Compute intersection between keep and nms_keep
    # keep = np.sets.intersection(np.expand_dims(keep, 0),
    #                                 np.expand_dims(nms_keep, 0))
    # keep = np.sparse.to_dense(keep)[0]
    # # Keep top detections
    # # roi_count = config['DETECTION_MAX_INSTANCES']
    # class_scores_keep = np.take(class_scores, keep)
    # # num_keep = np.minimum(np.shape(class_scores_keep)[0], roi_count)
    # top_ids = np.nn.top_k(class_scores_keep, k=np.shape(class_scores_keep)[0], sorted=True)[1]
    # keep = np.take(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    # detections = np.concatenate([
    #     np.take(refined_rois, keep),
    #     np.cast(np.take(class_ids, keep),dtype=np.float32)[..., np.newaxis],
    #     np.take(class_scores, keep)[..., np.newaxis]
    #     ], axis=1)
    
    pre_detections = [
        np.take(refined_rois, nms_keep, axis=0),
        np.take(pre_nms_class_ids, nms_keep),
        np.take(pre_nms_scores, nms_keep)]

    # print('pre_detections')
    # print(pre_detections[0].shape)
    # print(pre_detections[1].shape)
    # print(pre_detections[2].shape)

    assert len(pre_detections[0]) == len(nms_keep)
    detections = []
    for i in range(len(pre_detections[0])):
        detections.append(np.array([pre_detections[0][i][0],pre_detections[0][i][1],pre_detections[0][i][2],pre_detections[0][i][3],
            pre_detections[1][i],pre_detections[2][i],nms_keep[i]]))

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    # gap = config['DETECTION_MAX_INSTANCES'] - np.shape(detections)[0]
    # detections = np.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections

def conv_to_detections(rois,mrcnn_class,mrcnn_bbox,image_meta):

    config = {
        'IMAGES_PER_GPU' : 1,
        'BATCH_SIZE' : 1,
        'DETECTION_MAX_INSTANCES' : 100,
        'DETECTION_MIN_CONFIDENCE' : 0.7,
        'DETECTION_NMS_THRESHOLD' : 0.3,
        'BBOX_STD_DEV' : np.array([0.1, 0.1, 0.2, 0.2])
    }
    

    # Get windows of images in normalized coordinates. Windows are the area
    # in the image that excludes the padding.
    # Use the shape of the first image in the batch to normalize the window
    # because we know that all images get resized to the same size.

    m = parse_image_meta_graph(image_meta)
    image_shape = m['image_shape'][0]
    window = norm_boxes_graph(m['window'], image_shape[:2])

    # Run detection refinement graph on each item in the batch
    detections_batch = batch_slice(
        [rois, mrcnn_class, mrcnn_bbox, window],
        lambda x, y, w, z: refine_detections_graph(x, y, w, z, config),
        config['IMAGES_PER_GPU'])

    # Reshape output
    # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score, roi_index)] in
    # normalized coordinates
    return np.reshape(detections_batch,[config['BATCH_SIZE'], -1, 7])

def save_img(arr,fname,recshapes,oplague_masks=None):

    arr = np.array(arr,dtype=np.uint8)
    im = Image.fromarray(arr)
    im = im.convert('RGBA')
    width, height = im.size

    if oplague_masks is not None:
        assert oplague_masks.shape[0] == height
        assert oplague_masks.shape[1] == width

    img1 = ImageDraw.Draw(im)
    for rec in recshapes:
        assert len(rec) == 4
        img1.rectangle([rec[1],rec[0],rec[3],rec[2]], outline ="red")

    if oplague_masks is not None:
        assert len(oplague_masks.shape) == 3
        for row in range(oplague_masks.shape[0]):
            for col in range(oplague_masks.shape[1]):
                if len(oplague_masks[row][col]) == 1:
                    ret = oplague_masks[row][col]
                else:
                    ret = np.bitwise_or(*oplague_masks[row][col])
                if ret:
                    # Image lib uses x=col,y=row
                    rgba = list(im.getpixel((col,row)))
                    rgba[3] = 18
                    im.putpixel((col,row), tuple(rgba))
            
    im.save(fname+".png")

def single_img_test(data,test_generator,model,testconfig):
    ### try predict, but seems not working
    inputs,outputs = test_generator[0]
    sample_imgs = {
        'input_1' : np.array([inputs[0][0]]),
        'input_image_meta' : np.array([inputs[1][0]]), 
        'input_rpn_match' : np.array([inputs[2][0]]), 
        'input_rpn_bbox' : np.array([inputs[3][0]]), 
        'input_gt_class_ids' : np.array([inputs[4][0]]), 
        'input_gt_boxes' : np.array([inputs[5][0]]), 
        'input_gt_masks' : np.array([inputs[6][0]])
    }

    
    predictions = model.predict(sample_imgs,workers=1,use_multiprocessing=False)
    rpn_class_logits = predictions[0]
    rpn_class = predictions[1]
    rpn_bbox = predictions[2]
    mrcnn_class_logits = predictions[3]
    mrcnn_class = predictions[4]
    mrcnn_bbox = predictions[5]
    mrcnn_mask = predictions[6]
    rpn_rois = predictions[7]
    output_rois = predictions[8]
    rpn_class_loss = predictions[9]
    rpn_bbox_loss = predictions[10]
    mrcnn_class_loss = predictions[11]
    mrcnn_bbox_loss = predictions[12]
    mrcnn_mask_loss = predictions[13]
    
    # get image id
    img_id = int(sample_imgs['input_image_meta'][0][0])
    ori_img_arr = data.get_img(img_id)

    # save_img(ori_img_arr,"single_img_test",[])
    
    detections = conv_to_detections(output_rois,mrcnn_class,mrcnn_bbox,sample_imgs['input_image_meta'])
    
    # detections, _, _, mrcnn_mask, _, _, _ =\
    #     self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
    # Process detections
    ori_img_shape = np.array(inputs[1][0][1:4],dtype=np.uint32)
    mold_img_shape =  np.array(inputs[1][0][4:7],dtype=np.uint32)
    window = np.array(inputs[1][0][7:11],dtype=np.uint32)

    '''
    detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
    mrcnn_mask: [N, height, width, num_classes]
    original_image_shape: [H, W, C] Original image shape before resizing
    image_shape: [H, W, C] Shape of the image after resizing and padding
    window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
            image is excluding the padding.
    Returns:
    boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
    class_ids: [N] Integer class IDs for each bounding box
    scores: [N] Float probability scores of the class_id
    masks: [height, width, num_instances] Instance masks
    '''

    final_rois, final_class_ids, final_scores, final_masks =\
            unmold_detections(detections[0], mrcnn_mask[0], ori_img_shape, mold_img_shape,window)

    print("final_rois",final_rois.shape,final_rois)
    print("final_class_ids",final_class_ids.shape,final_class_ids)
    print("final_scores",final_scores.shape,final_scores)
    print("final_masks",final_masks.shape)

    save_img(ori_img_arr,"single_img_test_"+str(img_id),final_rois,final_masks)

    return
