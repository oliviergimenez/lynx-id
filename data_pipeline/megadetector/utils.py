from PIL import Image


def crop_bbox(row_bbox):
    img = Image.open(row_bbox["file"])
    x, y, width, height = row_bbox[['x', 'y', 'width', 'height']]

    im_width, im_height = img.size

    x_norm = x * im_width
    y_norm = y * im_height
    w_norm = width * im_width
    h_norm = height * im_height

    left = x_norm
    top = y_norm
    right = x_norm + w_norm
    bottom = y_norm + h_norm

    # Crop and save to visualize it
    return img.crop((left, top, right, bottom))
