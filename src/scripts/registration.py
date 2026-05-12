import numpy as np
import cv2
from skimage import transform as sktransform
from skimage.measure import ransac
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


########  ########  ######   ####  ######  ######## ########     ###    ######## ####  #######  ##    ##
##     ## ##       ##    ##   ##  ##    ##    ##    ##     ##   ## ##      ##     ##  ##     ## ###   ##
##     ## ##       ##         ##  ##          ##    ##     ##  ##   ##     ##     ##  ##     ## ####  ##
########  ######   ##   ####  ##   ######     ##    ########  ##     ##    ##     ##  ##     ## ## ## ##
##   ##   ##       ##    ##   ##        ##    ##    ##   ##   #########    ##     ##  ##     ## ##  ####
##    ##  ##       ##    ##   ##  ##    ##    ##    ##    ##  ##     ##    ##     ##  ##     ## ##   ###
##     ## ########  ######   ####  ######     ##    ##     ## ##     ##    ##    ####  #######  ##    ##


def _to_uint8(img):
    """Normalize to uint8 for ORB."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return (img * 255).astype(np.uint8)


def register_stacks(
    image_stack, ref_image_stack, transform_type="euclidean", return_details=False
):
    """
    Find a 2D transform mapping image_stack -> ref_image_stack,
    using features from every z-plane.

    Parameters
    ----------
    image_stack, ref_image_stack : ndarray, shape (Z, Y, X)
        Assumes plane z in moving corresponds to plane z in reference.
    transform_type : {'euclidean', 'affine'}

    Returns
    -------
    transform : skimage transform
        Maps moving coords -> reference coords.
        Use sktransform.warp(img, transform.inverse) to resample moving onto ref.
    """
    assert image_stack.shape == ref_image_stack.shape

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    src_all, dst_all, z_all = [], [], []
    for z in range(image_stack.shape[0]):
        mov = _to_uint8(image_stack[z])
        ref = _to_uint8(ref_image_stack[z])

        kp1, des1 = orb.detectAndCompute(mov, None)
        kp2, des2 = orb.detectAndCompute(ref, None)
        if des1 is None or des2 is None:
            continue

        matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
        matches = matches[: max(10, len(matches) // 2)]  # keep best half

        src_all.append(np.float32([kp1[m.queryIdx].pt for m in matches]))
        dst_all.append(np.float32([kp2[m.trainIdx].pt for m in matches]))
        z_all.append(np.array([z for m in matches]))

    src = np.vstack(src_all)
    dst = np.vstack(dst_all)
    zs = np.hstack(z_all)

    model_cls = {
        "euclidean": sktransform.EuclideanTransform,
        "affine": sktransform.AffineTransform,
    }[transform_type]

    model, inliers = ransac(
        (src, dst),
        model_cls,
        min_samples=3,
        residual_threshold=2.0,
        max_trials=2000,
    )
    # print(f"{inliers.sum()}/{len(inliers)} inliers across all planes")
    if return_details:
        return model, dict(src=src, dst=dst, inliers=inliers, zs=zs)
    else:
        return model


def apply_transform(image_stack, transform):
    """Warp every plane of a stack using the same 2D transform."""
    out = np.empty_like(image_stack, dtype=np.float32)
    for z in range(image_stack.shape[0]):
        out[z] = sktransform.warp(
            image_stack[z],
            transform.inverse,
            preserve_range=True,
            order=1,
        )
    return out


######## ##     ##    ###    ##
##       ##     ##   ## ##   ##
##       ##     ##  ##   ##  ##
######   ##     ## ##     ## ##
##        ##   ##  ######### ##
##         ## ##   ##     ## ##
########    ###    ##     ## ########


def evaluate(ref_stack, moving_stack, mask=None):
    """Per-plane NCC. Pass mask to exclude warp borders."""
    ncc = []
    for z in range(ref_stack.shape[0]):
        r, m = ref_stack[z], moving_stack[z]
        if mask is not None:
            r, m = r[mask[z]], m[mask[z]]
        ncc.append(pearsonr(r.ravel(), m.ravel())[0])

    return np.array(ncc)


########  ##        #######  ########
##     ## ##       ##     ##    ##
##     ## ##       ##     ##    ##
########  ##       ##     ##    ##
##        ##       ##     ##    ##
##        ##       ##     ##    ##
##        ########  #######     ##


def _norm(img):
    """Normalize a 2D image to [0, 1] using 1/99 percentiles."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def _green_magenta(ref, mov):
    """Composite: ref -> green channel, mov -> red+blue (magenta)."""
    rgb = np.zeros((*ref.shape, 3), dtype=np.float32)
    rgb[..., 0] = mov  # R
    rgb[..., 1] = ref  # G
    rgb[..., 2] = mov  # B
    return rgb


def inspect_registration_delta(
    image_stack,
    reference_stack,
    transformed_stack,
    z,
    interval=1000,
    save_path=None,
    frames_per_second=1,
    figsize=(15, 10),
):
    assert reference_stack.shape == image_stack.shape == transformed_stack.shape

    moving = _norm(image_stack[z])
    reference = _norm(reference_stack[z])
    transformed = _norm(transformed_stack[z])

    fig, axes = plt.subplots(3, 3, figsize=figsize, constrained_layout=True)
    (
        (axis_top_left, axis_top_middle, axis_top_right),
        (axis_middle_left, axis_middle_middle, axis_middle_right),
        (axis_bottom_left, axis_bottom_middle, axis_bottom_right),
    ) = axes

    # the composite images
    axis_top_left.matshow(_green_magenta(reference, moving))
    axis_top_left.set_title("Before: reference (green) / moving (magenta)")

    axis_middle_left.matshow(_green_magenta(reference, transformed))
    axis_middle_left.set_title("After: reference (green) / transformed (magenta)")

    axis_bottom_left.matshow(_green_magenta(moving, transformed))
    axis_bottom_left.set_title("After: moving (green) / transformed (magenta)")

    # the animations
    image_top_middle = axis_top_middle.matshow(reference, cmap="gray", vmin=0, vmax=1)
    image_middle_middle = axis_middle_middle.matshow(
        reference, cmap="gray", vmin=0, vmax=1
    )
    image_bottom_middle = axis_bottom_middle.matshow(
        moving, cmap="gray", vmin=0, vmax=1
    )

    def update(frame):
        show_reference = frame % 2 == 0
        if show_reference:
            image_top_middle.set_data(reference)
            image_middle_middle.set_data(reference)
            image_bottom_middle.set_data(moving)
            axis_top_middle.set_title("reference")
            axis_middle_middle.set_title("reference")
            axis_bottom_middle.set_title("moving")
        else:
            image_top_middle.set_data(moving)
            image_middle_middle.set_data(transformed)
            image_bottom_middle.set_data(transformed)
            axis_top_middle.set_title("moving")
            axis_middle_middle.set_title("transformed")
            axis_bottom_middle.set_title("transformed")
        return (
            image_top_middle,
            image_bottom_middle,
            image_bottom_middle,
            axis_top_middle,
            axis_middle_middle,
            axis_bottom_middle,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=2,
        interval=interval,
        blit=False,
        repeat=True,
    )

    # the difference images
    difference_before = reference - moving  # alignment error before
    difference_after = reference - transformed  # alignment error after
    difference_transform = transformed - moving  # changes induced by the transform

    # # Independent symmetric scales — the two panels measure different things.
    max_absolute_before = max(np.abs(difference_before).max(), 1e-8)

    image_top_right = axis_top_right.matshow(
        difference_before,
        cmap="RdBu_r",
        vmin=-max_absolute_before,
        vmax=max_absolute_before,
    )
    nccs = evaluate(reference_stack, image_stack)
    title = f"diff - ncc plane:{nccs[z]:.2f} - ncc mean:{np.average(nccs):.2f}"
    axis_top_right.set_title(title)

    image_middle_right = axis_middle_right.matshow(
        difference_after,
        cmap="RdBu_r",
        vmin=-max_absolute_before,
        vmax=max_absolute_before,
    )
    nccs = evaluate(reference_stack, transformed_stack)
    title = f"diff - ncc plane:{nccs[z]:.2f} - ncc mean:{np.average(nccs):.2f}"
    axis_middle_right.set_title(title)

    image_bottom_right = axis_bottom_right.matshow(
        difference_transform,
        cmap="RdBu_r",
        vmin=-max_absolute_before,
        vmax=max_absolute_before,
    )
    axis_bottom_right.set_title("diff")

    fig.colorbar(image_top_right, ax=axis_top_right, fraction=0.046, pad=0.02)
    fig.colorbar(image_middle_right, ax=axis_middle_right, fraction=0.046, pad=0.02)
    fig.colorbar(image_bottom_right, ax=axis_bottom_right, fraction=0.046, pad=0.02)

    for ax in axes.flatten():
        ax.axis("off")
        if ax.get_label() == "<colorbar>":
            continue

    if save_path is not None:
        animation.save(save_path, writer=PillowWriter(fps=frames_per_second))
        print(f"Saved animation to {save_path}")

    return animation


def plot_keypoints(img_data, reg_details, z, save_path=False):
    fig, axes = plt.subplots()
    img_cat = np.concatenate([img_data["stack"][z], img_data["aligned"][z]], axis=1)
    img_kwargs = dict(
        vmin=np.percentile(img_cat, 5), vmax=np.percentile(img_cat, 99), cmap="gray"
    )
    axes.matshow(img_cat, **img_kwargs)

    # lines between keypoints
    ix = np.logical_and(reg_details["zs"] == z, reg_details["inliers"])
    src = reg_details["src"][ix]
    dst = reg_details["dst"][ix]
    offset = img_data["stack"].shape[2]
    import seaborn as sns

    colors = sns.color_palette("tab10", n_colors=ix.shape[0])
    for i in range(src.shape[0]):
        axes.plot(
            *np.vstack([src[i], dst[i] + np.array([offset, 0])]).T, lw=0.6, c=colors[i]
        )
    # lines between src and dst
    axes.set_axis_off()

    if save_path is not None:
        fig.savefig(save_path.with_suffix(".png"), dpi=300)
