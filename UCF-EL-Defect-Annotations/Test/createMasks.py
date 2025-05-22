# changelog
# 05-21-2025 Dawn
#   Updated color cmap
#   added color overlay function in color_overlay folder
#   fixed bugs -> switch to dataframes to do subdata frames on each image annotation
#   added comments
#   proposed fixes/updates for annotation styles
#   Other Notes:
#       since VGG is open source, maybe look into combining both into single application or forking it specifically for colors



import os
import json
import cv2
from matplotlib import category
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd         # added pandas for dataframe use ing 




# Define defect categories and their colors
defect_categories = {
    'Contact_FrontGridInterruption': 1,
    'Contact_NearSolderPad': 1,

    'Contact_BeltMarks': 4,
    'Contact_Corrosion': 4,

    'Interconnect_Disconnected': 3,
    'Interconnect_HighlyResistive': 3,
    'Interconnect_BrightSpot': 3,

    'Crack_Closed': 2,
    'Crack_Resistive': 2,
    'Crack_Isolated': 2
}


"""
        if shape_attributes['name'] == 'rect':
            mask = create_mask_from_rectangles([shape_attributes], width, height)
        else:
            mask = create_mask_from_polygons([shape_attributes], width, height)
                
        # apply defect-specific color
        colored_mask = create_colored_mask(defect_class, width, height)
        colored_mask[mask == 255] = colored_mask[mask == 255] * 0.5
                
        # combine with existing mask
        combined_mask = np.maximum(combined_mask, mask)
            
        # save binary mask
        mask_path = os.path.join(masks_dir, f"{path(filename).stem}_mask.png")
        cv2.imwrite(mask_path, combined_mask)
            
        # create and save overlay
        overlay = image.copy().astype(np.float32) / 255.0
        overlay[combined_mask == 255] = overlay[combined_mask == 255] * 0.5
            
        overlay_path = os.path.join(overlays_dir, f"{path(filename).stem}_overlay.png")
        cv2.imwrite(overlay_path, overlay * 255.0)
"""

#updated with one in the model 
# create custom colormap for image visualizations [Black, Red, Blue, Purple, Orange]
cmaplist = [(0.001462, 0.000466, 0.013866, 1.0),                                    
            (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0),    
            (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0),    
            (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0),       
            (1.0, 0.4980392156862745, 0.0, 1.0)]                                    




def parse_region_attributes(region_str: str) -> dict:
    """Parse region attributes from JSON string."""
    return json.loads(region_str)

def create_mask_from_rectangles(rectangles: List[dict], width: int, height: int) -> np.ndarray:
    """Create mask from rectangle annotations."""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for rect in rectangles:
        x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    
    return mask

def create_mask_from_polygons(polygons: List[dict], width: int, height: int) -> np.ndarray:
    """Create mask from polygon annotations."""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for poly in polygons:
        points = np.array(list(zip(poly['all_points_x'], poly['all_points_y'])))

        cv2.fillPoly(mask, [points], 255)
    
    return mask
# changed a few things here for using json structure instead of a (String)
def create_colored_mask(defect_json, width: int, height: int)-> np.ndarray:
    """Create colored mask based on defect class."""
    category_index = defect_categories.get(defect_json["Defect_Class"])
    color = cmaplist[category_index][:3]  # Remove alpha channel
    return np.full((height, width, 3), color, dtype=np.float32)


# method for taking inth % avlues of (r, g, b) and turns into int(B, G, R) for cv2 things
def convertRGB_BGR(color):
    color = np.array(color)
    color = color * 255
    color = tuple(color)
    color = color[::-1]
    return color

# grabbed from Dawn's OCR debug system and modified for the attribute color choosing 
def annotate_image_cv2(img_path, regions, region_attributes, out_path=None, fill_alpha=0.3):
    img = cv2.imread(img_path)
    overlay = img.copy()

    for reg, attribute, in zip (regions, region_attributes):
        #turn into normal jsons
        reg = parse_region_attributes(reg)
        attribute = parse_region_attributes(attribute)
        # get category index for color purposing
        category_index = defect_categories.get(attribute["Defect_Class"])
        
        # get the colors for the annotations
        color = cmaplist[category_index][:3]
        color = convertRGB_BGR(color)
        
        #skips for non polygons, see bottom of code for potential solutions for all annotation region types
        if reg["name"] != "polygon":
            continue
        points = np.array(list(zip(reg["all_points_x"], reg["all_points_y"])), dtype=np.int32).reshape(-1,1,2)


        cv2.polylines(img, [points], True, color, 1)
        cv2.fillPoly(overlay, [points], color)

    #combines to take the right path
    cv2.addWeighted(overlay, fill_alpha, img, 1-fill_alpha, 0, img)
    if out_path:
        cv2.imwrite(out_path, img)    
    return img

def create_dataframe(csv_path):
    df = pd.read_csv(csv_path, header=0)
    return df

def process_annotations(csv_path: str, images_dir: str, output_dir: str):
    """
    Process annotations and create masks with colored overlays.
    
    Args:
        csv_path: Path to CSV file containing annotations
        images_dir: Directory containing original images
        output_dir: Output directory for masks and overlays
    """
    # Create output directories and paths
    masks_dir = os.path.join(output_dir, 'masks')
    overlays_dir = os.path.join(output_dir, 'overlays')
    color_overlays_dir = os.path.join(output_dir, 'color_overlays')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(color_overlays_dir, exist_ok=True)

    # changed the system from trying to parse via just ',' to using dataframes and subdata frames for different images and their datasets
    # it might be good to create a function to output/read the subdata frames at a later date 
    # also look into what the previous iterations on the code were and their structure system. 
    data_df = create_dataframe(csv_path)
    image_data_frames = [group for _, group in data_df.groupby("filename")]

    for image_file_df in image_data_frames:
        # get the file name
        filename = image_file_df.iloc[0, 0]
        # get the attributes and datas
        shape_attributes = image_file_df["region_shape_attributes"].to_list()
        region_attributes = image_file_df["region_attributes"].to_list()
        #create base image path
        image_path =  os.path.join(images_dir, filename)


        # this was added by dawn to do a color creation, as that wasn't entirely clear with the code structure
        #get color system
        color_overlays_path = os.path.join(color_overlays_dir, f"{filename}_overlay.png")
        #send to be color annotated
        # this will create a color in the color directory
        annotate_image_cv2(image_path, regions=shape_attributes, region_attributes=region_attributes, out_path=color_overlays_path)



        #lets try and fix the one joeseph did:
        if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
        # Load original image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        combined_mask = np.zeros((height, width), dtype=np.uint8)
        

        for region, attribute, in zip (shape_attributes, region_attributes):
            shape = parse_region_attributes(region)
            defect = parse_region_attributes(attribute)
            # get category index for color purposing
            category_index = defect_categories.get(defect["Defect_Class"])

            if shape['name'] == 'rect':
                mask = create_mask_from_rectangles([shape], width, height)
            else:
                mask = create_mask_from_polygons([shape], width, height)


            # not sure if needed or properly used. but i think the idea was to make a full color mask, and then apply that to the built mask?
            # you would need to use cv2.addWeighted as done in annotate_image_cv2()
            colored_mask = create_colored_mask(defect, width, height)                        
            colored_mask[mask == 255] = colored_mask[mask == 255] * 0.5

            # Combine with existing mask
            combined_mask = np.maximum(combined_mask, mask)

        #process each region:
        # Save binary mask
        # this binary mask is just the black and white outlines of what defects/where annotations were.
        mask_path = os.path.join(masks_dir, f"{Path(filename).stem}_mask.png")
        cv2.imwrite(mask_path, combined_mask)
            
        # Create and save overlay 
        overlay = image.copy().astype(np.float32) / 255.0
        overlay[combined_mask == 255] = overlay[combined_mask == 255] * 0.5
        
        #this is throwing a fallback warning with imwrite
        #           Unsupported depth image for selected encoder is fallbacked to CV_8U.
        overlay_path = os.path.join(overlays_dir, f"{Path(filename).stem}_overlay.png")
        cv2.imwrite(overlay_path, overlay * 255.0)


def main():
    
    """Example usage."""
    csv_path = r"path"
    images_dir = r"ptah"
    output_dir = r"path"
    
    process_annotations(csv_path, images_dir, output_dir)

if __name__ == "__main__":
    main()

#notation for poential other annotation ideas
"""
    # potential use cases for using the other annotation drawings, will need to debug for further use


    # name = name = reg.get("name")
    # if name == "rect":
    #     x, y = int(reg["x"]), int(reg["y"])
    #     w, h = int(reg["width"]), int(reg["height"])
    #     pt1, pt2 = (x, y), (x + w, y + h)
    #     cv2.rectangle(img, pt1, pt2, color, thickness)
    #     if show_fill:
    #         cv2.rectangle(overlay, pt1, pt2, color, -1)

    # elif name == "circle":
    #     center = (int(reg["cx"]), int(reg["cy"]))
    #     radius = int(reg["r"])
    #     cv2.circle(img, center, radius, color, thickness)
    #     if show_fill:
    #         cv2.circle(overlay, center, radius, color, -1)

    # elif name == "ellipse":
    #     center = (int(reg["cx"]), int(reg["cy"]))
    #     axes = (int(reg["rx"]), int(reg["ry"]))
    #     angle = float(reg.get("theta", 0))
    #     cv2.ellipse(img, center, axes, angle, 0, 360, color, thickness)
    #     if show_fill:
    #         cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1)

    # elif name == "polygon":
    #     pts = np.array(list(zip(reg["all_points_x"], reg["all_points_y"])),
    #                    dtype=np.int32).reshape(-1,1,2)
    #     cv2.polylines(img, [pts], True, color, thickness)
    #     if show_fill:
    #         cv2.fillPoly(overlay, [pts], color)

    # elif name == "point":
    #     center = (int(reg["cx"]), int(reg["cy"]))
    #     cv2.circle(img, center, thickness*2, color, -1)

    # else:
    #     # unsupported shape
    #     continue"""
