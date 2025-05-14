import cv2
import numpy as np

class StitcherPanorama:
    def __init__(self, feature_detector='ORB', nfeatures=2000):
        # Initialize ORB detector for faster feature extraction
        self.detector = cv2.ORB_create(nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_describe(self, image):
        """Detect keypoints and compute ORB descriptors."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        if kp is None or des is None:
            return None, None
        # Convert keypoints to NumPy float32 array of (x, y) coordinates
        pts = np.array([kp_i.pt for kp_i in kp], dtype=np.float32)
        return pts, des

    def match_keypoints(self, ptsA, ptsB, desA, desB, ratio=0.75, reprojThresh=5.0):
        """
        Match keypoints using the ratio test and estimate an affine transform.
        Returns the 2x3 affine matrix if successful, otherwise None.
        """
        if desA is None or desB is None:
            return None

        matches = self.matcher.knnMatch(desA, desB, k=2)
        good_matches = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < ratio * n.distance:
                good_matches.append(m)

        # Need at least 4 matches to compute affine
        if len(good_matches) < 4:
            return None

        src_pts = np.float32([ptsA[m.queryIdx] for m in good_matches])
        dst_pts = np.float32([ptsB[m.trainIdx] for m in good_matches])
        M, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=reprojThresh)
        return M

    def stitch_right(self, panorama, new_image):
        """
        Stitch a new image onto the right side of the current panorama.
        Returns the updated panorama.
        """
        # Detect and match features
        ptsA, desA = self.detect_and_describe(new_image)
        ptsB, desB = self.detect_and_describe(panorama)
        M = self.match_keypoints(ptsA, ptsB, desA, desB)
        if M is None:
            # If matching fails, return the original panorama
            return panorama

        # Calculate dimensions for the stitched image
        h_pan, w_pan = panorama.shape[:2]
        h_new, w_new = new_image.shape[:2]
        h_total = max(h_pan, h_new)
        w_total = w_pan + w_new

        # Warp the new image onto a canvas of width w_total
        warped_new = np.zeros((h_total, w_total, 3), dtype=panorama.dtype)
        cv2.warpAffine(new_image, M, (w_total, h_total), warped_new,
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        # Prepare a canvas for the existing panorama
        pano_canvas = np.zeros((h_total, w_total, 3), dtype=panorama.dtype)
        pano_canvas[:h_pan, :w_pan] = panorama

        # Create linear blending masks to smoothly merge the overlap
        blend_width = 50
        seam = w_pan
        mask_pano = np.zeros((h_total, w_total), dtype=np.float32)
        mask_new = np.zeros((h_total, w_total), dtype=np.float32)

        # Panorama mask: solid 1 up to the start of the blend region
        mask_pano[:, :seam - blend_width] = 1.0
        for i in range(blend_width * 2):
            alpha = 1.0 - i / (blend_width * 2)
            x = seam - blend_width + i
            if 0 <= x < w_total:
                mask_pano[:, x] = alpha

        # New image mask: ramp up from 0 to 1 across the blend region
        for i in range(blend_width * 2):
            alpha = i / (blend_width * 2)
            x = seam - blend_width + i
            if 0 <= x < w_total:
                mask_new[:, x] = alpha
        mask_new[:, seam + blend_width:] = 1.0

        # Merge masks into 3 channels
        mask_pano = cv2.merge([mask_pano]*3)
        mask_new = cv2.merge([mask_new]*3)

        # Blend the two images using the masks
        blended = (pano_canvas.astype(np.float32) * mask_pano +
                   warped_new.astype(np.float32) * mask_new)
        blended = blended.astype(np.uint8)

        # Crop out any black borders and return
        return self.crop_black(blended)

    def stitch_left(self, panorama, new_image):
        """
        Stitch a new image onto the left side of the current panorama.
        Returns the updated panorama.
        """
        # Calculate dimensions for padding
        h_pan, w_pan = panorama.shape[:2]
        h_new, w_new = new_image.shape[:2]
        h_total = max(h_pan, h_new)
        w_total = w_new + w_pan

        # Prepare a canvas with panorama shifted right by w_new (padding on left)
        pano_canvas = np.zeros((h_total, w_total, 3), dtype=panorama.dtype)
        pano_canvas[:h_pan, w_new:w_new + w_pan] = panorama

        # Detect and match features between new_image and padded panorama
        ptsA, desA = self.detect_and_describe(new_image)
        ptsB, desB = self.detect_and_describe(pano_canvas)
        M = self.match_keypoints(ptsA, ptsB, desA, desB)
        if M is None:
            return panorama

        # Warp the new image onto the padded canvas
        warped_new = np.zeros((h_total, w_total, 3), dtype=panorama.dtype)
        cv2.warpAffine(new_image, M, (w_total, h_total), warped_new,
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        # Create blending masks
        blend_width = 50
        seam = w_new
        mask_pano = np.zeros((h_total, w_total), dtype=np.float32)
        mask_new = np.zeros((h_total, w_total), dtype=np.float32)

        # New image mask: solid 1 on its left side
        mask_new[:, :seam - blend_width] = 1.0
        for i in range(blend_width * 2):
            alpha = 1.0 - i / (blend_width * 2)
            x = seam - blend_width + i
            if 0 <= x < w_total:
                mask_new[:, x] = alpha

        # Panorama mask: ramp up from 0 to 1 across the blend region
        for i in range(blend_width * 2):
            alpha = i / (blend_width * 2)
            x = seam - blend_width + i
            if 0 <= x < w_total:
                mask_pano[:, x] = alpha
        mask_pano[:, seam + blend_width:] = 1.0

        # Merge masks into 3 channels
        mask_pano = cv2.merge([mask_pano]*3)
        mask_new = cv2.merge([mask_new]*3)

        # Blend and combine
        blended = (warped_new.astype(np.float32) * mask_new +
                   pano_canvas.astype(np.float32) * mask_pano)
        blended = blended.astype(np.uint8)
        return self.crop_black(blended)

    def crop_black(self, image):
        """
        Crop out black borders from the final panorama.
        Looks for any non-zero pixels and trims the image to that bounding box.
        """
        if len(image.shape) == 3:
            # Consider any of the 3 channels
            mask = (image[:, :, 0] > 0) | (image[:, :, 1] > 0) | (image[:, :, 2] > 0)
        else:
            mask = image > 0

        if not np.any(mask):
            return image  # return as is if no non-black pixels

        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return image[y0:y1, x0:x1]

def read_frames(video_path):
    """Read all frames from a video file and resize each frame."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to width=720, height=1080 for consistency
        resized = cv2.resize(frame, (720, 1080))
        frames.append(resized)
    cap.release()
    return frames

def cylindrical_warp(image):
    """
    Apply cylindrical projection to the image to simulate wide-angle camera.
    Uses a fixed focal length for warping.
    """
    h, w = image.shape[:2]
    # Mock camera intrinsics (focal length ~1030, center at image center)
    K = np.array([[1030,    0, w / 2],
                  [0,    1030, h / 2],
                  [0,       0,     1]])
    Kinv = np.linalg.inv(K)

    # Pixel coordinates
    y_i, x_i = np.indices((h, w))
    homogeneous = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(-1, 3)
    norm = Kinv.dot(homogeneous.T).T
    x = norm[:, 0]
    y = norm[:, 1]
    z = norm[:, 2]

    # Project onto cylinder
    X = np.stack([np.sin(x), y, np.cos(x)], axis=-1)
    projected = K.dot(X.T).T
    projected = projected[:, :2] / projected[:, 2:3]

    # Map back to image
    map_x = projected[:, 0].reshape(h, w).astype(np.float32)
    map_y = projected[:, 1].reshape(h, w).astype(np.float32)
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return warped

def resize_frame(frame):
    """
    Scale the frame by 1.1x and then center-crop it back to the original size.
    Helps to remove black edges introduced by cylindrical warp.
    """
    h, w = frame.shape[:2]
    new_h, new_w = int(h * 1.1), int(w * 1.1)
    resized = cv2.resize(frame, (new_w, new_h))
    start_y = (new_h - h) // 2
    start_x = (new_w - w) // 2
    cropped = resized[start_y:start_y + h, start_x:start_x + w]
    return cropped

def cut_corners(panorama):
    """
    Crop away a small percentage of the border of the panorama to remove artifacts.
    """
    h, w = panorama.shape[:2]
    percent = 0.08
    h_crop = int(h * (1 - 2 * percent))
    w_crop = int(w * (1 - 2 * percent))
    start_y = int(h * percent)
    start_x = int(w * percent)
    return panorama[start_y:start_y + h_crop, start_x:start_x + w_crop]

def generate_pano(video_path, output_path):
    """
    Main function to generate a panorama from a video.
    Displays progress for each stitched frame.
    """
    print(f"Loading video: {video_path}")
    frames = read_frames(video_path)
    n = len(frames)
    if n == 0:
        print("  [Error] No frames found.")
        return

    # Pre–warp and resize all frames
    warped = []
    for idx, f in enumerate(frames, 1):
        cyl = cylindrical_warp(f)
        cyl = resize_frame(cyl)
        warped.append(cyl)
        print(f"  [Preprocess] frame {idx}/{n}")

    mid = n // 2
    stitcher = StitcherPanorama()
    panorama = warped[mid]

    total_steps = n
    step = 1
    print(f"[Stitch] Step {step}/{total_steps}: initialize with frame {mid}")

    offset = 1
    while (mid - offset) >= 0 or (mid + offset) < n:
        if (mid - offset) >= 0:
            step += 1
            print(f"[Stitch] Step {step}/{total_steps}: stitch left frame {mid - offset}")
            panorama = stitcher.stitch_left(panorama, warped[mid - offset])
        if (mid + offset) < n:
            step += 1
            print(f"[Stitch] Step {step}/{total_steps}: stitch right frame {mid + offset}")
            panorama = stitcher.stitch_right(panorama, warped[mid + offset])
        offset += 1

    print("Cropping final panorama...")
    panorama = cut_corners(panorama)
    cv2.imwrite(output_path, panorama)
    print(f"Done. Panorama saved to {output_path}")

