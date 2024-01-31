from PIL import Image


def crop_bbox(row_bbox, to_be_normalized=False):
    img = Image.open(row_bbox["file"])
    x, y, width, height = row_bbox[['x', 'y', 'width', 'height']]

    if to_be_normalized:
        im_width, im_height = img.size

        x *= im_width
        y *= im_height
        width *= im_width
        height *= im_height

    left = x
    top = y
    right = x + width
    bottom = y + height

    # Crop and save to visualize it
    return img.crop((left, top, right, bottom))
