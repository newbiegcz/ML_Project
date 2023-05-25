
from imgui.integrations.glfw import GlfwRenderer
import threading
import OpenGL.GL as gl
import glfw
import imgui
import sys
import time
import rich
import torch
import numpy as np
import data.dataset as dataset
import third_party.imgui_datascience.imgui_cv as imgui_cv
import matplotlib.pyplot as plt
import cv2

initialized = False
width, height = 1280, 720

alpha_gt = 0.3
alpha_pred = 0.3

cm_20c = plt.get_cmap('tab20c')
cm_20b = plt.get_cmap('tab20b')
colors = np.concatenate([cm_20c.colors, cm_20b.colors], axis=0)

current_object = None
objects = []

imgui_cv.USE_FAST_HASH = True

class DisplaySettings:
    def __init__(self, len_label, data_type = None):
        assert data_type in ["2D", "3D"], "data_type must be either 2D or 3D"
        self.current_height = 0
        self.len_label = len_label
        self.show_gt = np.ones((len_label,), dtype=np.int8)
        self.show_pred = np.ones((len_label,), dtype=np.int8)
        self.show_gt[current_object.data["background_label"]] = 0
        self.show_pred[current_object.data["background_label"]] = 0
        self.show_prompt = True

    def __eq__(self, other):
        return self.current_height == other.current_height and \
            (self.show_gt == other.show_gt).all() and \
            (self.show_pred == other.show_pred).all() and \
            self.len_label == other.len_label and \
            self.show_prompt == other.show_prompt
    
    def as_dict(self):
        return {
            "current_height": self.current_height,
            "show_gt": self.show_gt,
            "show_pred": self.show_pred,
            "len_label": self.len_label,
            "show_prompt": self.show_prompt,
        }

display_settings = None

class Object:
    def __init__(self, name, data_type, data):
        self.name = name
        self.data_type = data_type
        self.data = data

def impl_glfw_init():
    # IMGUI 这边似乎都以 Window 上 Pixel 为单位
    window_name = "Visualizer"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        sys.exit(1)

    return window

class Vec2:
    def __init__(self, vec2):
        self.x = vec2.x
        self.y = vec2.y
    def __add__(self, other):
        return Vec2(imgui.Vec2(self.x + other.x, self.y + other.y))
    def __sub__(self, other):
        return Vec2(imgui.Vec2(self.x - other.x, self.y - other.y))
    
last_args_dict = None
cached_output_image = None
dice_data = None

def copy_np_dict(x):
    if isinstance(x, dict):
        return {k: copy_np_dict(v) for k, v in x.items()}
    elif isinstance(x, np.ndarray):
        return x.copy()
    else:
        return x
    
def neq_np_dict(x, y):
    if isinstance(x, dict):
        if not isinstance(y, dict):
            return True
        if len(x) != len(y):
            return True
        for k, v in x.items():
            if k not in y:
                return True
            if neq_np_dict(v, y[k]):
                return True
        return False
    elif isinstance(x, np.ndarray):
        if not isinstance(y, np.ndarray):
            return True
        return not (x == y).all()
    else:
        return x != y

def pre_display_func_2d(image, pd_label, gt_label, prompt_points, label_name, background_label):
    global last_args_dict
    global cached_output_image
    global dice_data
    global display_settings

    args_dict = locals().copy() # must be the first line
    args_dict['display_settings'] = display_settings.as_dict()
    if last_args_dict is None or neq_np_dict(last_args_dict, args_dict):
        last_args_dict = copy_np_dict(args_dict)

        image = image.transpose(1, 2, 0)
        if image.shape[2] == 1:
            image = np.concatenate([image, image, image], axis=2)

        if display_settings.show_prompt:
            if prompt_points is not None:
                for point, t in prompt_points:
                    color = (0, 0, 1.0) if t else (1.0, 0, 0) # OpenCV: BGR
                    image = cv2.circle(image.copy(), point, 10, color, -1)

        for i in range(len(label_name)):
            if display_settings.show_gt[i]:
                t = gt_label == i
                image[t] = image[t] * (1 - alpha_gt) + np.array([[colors[2 * i]]]) * alpha_gt
            
        for i in range(len(label_name)):
            if display_settings.show_pred[i]:
                t = pd_label == i
                image[t] = image[t] * (1 - alpha_pred) + np.array([[colors[2 * i + 1]]]) * alpha_pred

        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)

        if pd_label is None or gt_label is None:
            dice_data = None
        else :
            dice_data = np.zeros((len(label_name),), dtype=np.float32)

            for i in range(len(label_name)):
                TP = np.sum(np.logical_and(pd_label == i, gt_label == i))
                FP = np.sum(np.logical_and(pd_label == i, gt_label != i))
                FN = np.sum(np.logical_and(pd_label != i, gt_label == i))
                if 2 * TP + FP + FN == 0 or i == background_label:
                    dice_data[i] = np.nan
                else :
                    dice_data[i] = 2 * TP / (2 * TP + FP + FN)
        # convert to bgr
        cached_output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cached_output_image

def main_loop():
    global display_settings
    global initialized
    global current_object
    global objects
    imgui.create_context()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)

    initialized = True

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        if not current_object is None:
            if current_object.data_type == "2D":
                pre_display_func_2d(current_object.data['image'], current_object.data['pd_label'], current_object.data['gt_label'], current_object.data['prompt_points'], current_object.data['label_name'], current_object.data['background_label'])
            elif current_object.data_type == "3D":
                image = current_object.data['image'][display_settings.current_height]
                if current_object.data['pd_label'] is not None:
                    pd_label = current_object.data['pd_label'][display_settings.current_height]
                else:
                    pd_label = None
                if current_object.data['gt_label'] is not None:
                    gt_label = current_object.data['gt_label'][display_settings.current_height]
                else:
                    gt_label = None
                if current_object.data['prompt_points'] is not None:
                    prompt_points = current_object.data['prompt_points'][display_settings.current_height]
                else:
                    prompt_points = None
                pre_display_func_2d(image, pd_label, gt_label, prompt_points, current_object.data['label_name'], current_object.data['background_label'])
            else:
                raise Exception("Unknown data type")

        imgui.new_frame()
        # create a full screen window
        imgui.set_next_window_size(width, height)
        imgui.set_next_window_position(0, 0)
        imgui.begin("Visualizer", False, imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR)

        left_pane_width = 500
        
        imgui.begin_child("left pane", width=left_pane_width, border=True, flags=imgui.WINDOW_NO_SCROLLBAR)

        with imgui.begin_group():
            with imgui.begin_child("Object", height=height - height // 3, border=True):
                if current_object is None:
                    imgui.text("No object has been selected.")
                else :
                    imgui.text("Current Object: %s (%s)" % (current_object.name, current_object.data_type))
                    if current_object.data_type == "2D":
                        pass
                    elif current_object.data_type == "3D":
                        # add a slider to select the height
                        changed, display_settings.current_height = imgui.slider_int("Height", display_settings.current_height, 0, current_object.data['image'].shape[0] - 1)
                    else:
                        raise Exception("Unknown data type")
                    
                    clicked, _ = imgui.checkbox("Show Prompt", display_settings.show_prompt)
                    if clicked:
                        display_settings.show_prompt = not display_settings.show_prompt
                    
                    with imgui.begin_table("table", 4):
                        imgui.table_setup_column("Name")
                        imgui.table_setup_column("GT")
                        imgui.table_setup_column("PD")
                        imgui.table_setup_column("Dice")

                        imgui.table_headers_row()
                        
                        for i, label in enumerate(current_object.data["label_name"]):
                            imgui.table_next_row()
                            imgui.table_next_column()
                            imgui.text(label)
                            imgui.table_next_column()
                            imgui.push_style_color(imgui.COLOR_CHECK_MARK, *colors[2*i], 1.0)
                            
                            clicked, _ = imgui.checkbox("##check_%d"%i, display_settings.show_gt[i] == 1)
                            imgui.pop_style_color()
                            if clicked:
                                display_settings.show_gt[i] = int ((display_settings.show_gt[i] + clicked) == 1)
                            imgui.table_next_column()
                            imgui.push_style_color(imgui.COLOR_CHECK_MARK, *colors[2*i+1], 1.0)
                            clicked, _ = imgui.checkbox("##check_%d_"%i, display_settings.show_pred[i] == 1)
                            imgui.pop_style_color()
                            if clicked:
                                display_settings.show_pred[i] = int ((display_settings.show_pred[i] + clicked) == 1)
                            imgui.table_next_column()
                            if i == current_object.data['background_label']:
                                imgui.text("Ignored")
                            else:
                                if dice_data is not None and dice_data[i] == dice_data[i]: # check nan
                                    imgui.text("%.3f" % dice_data[i])
                                else :
                                    imgui.text("N/A")
                        imgui.table_next_row()
                        imgui.table_next_column()
                        imgui.text("All")
                        imgui.table_next_column()
                        clicked, _ = imgui.checkbox("##check_all_gt", display_settings.show_gt.all() == 1)
                        if clicked:
                            if not display_settings.show_gt.all():
                                display_settings.show_gt = np.ones(len(current_object.data["label_name"]))
                            else:
                                display_settings.show_gt = np.zeros(len(current_object.data["label_name"]))

                        imgui.table_next_column()
                        clicked, _ = imgui.checkbox("##check_all_pred", display_settings.show_pred.all() == 1)
                        if clicked:
                            if not display_settings.show_pred.all():
                                display_settings.show_pred = np.ones(len(current_object.data["label_name"]))
                            else:
                                display_settings.show_pred = np.zeros(len(current_object.data["label_name"]))
                        imgui.table_next_column()
                        if dice_data is not None:
                            mean = np.nanmean(dice_data)
                            if mean == mean:
                                imgui.text("%.3f" % mean)
                            else :
                                imgui.text("N/A")
                        else :
                            imgui.text("N/A")

            
            with imgui.begin_child("Selector", border=True):
                with imgui.begin_child("Selector_Scroll", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR):
                    for i, obj in enumerate(objects):
                        opened, _ = imgui.selectable(obj.name + "##selectable_" + obj.name +"%d"%i, obj == current_object)
                        if opened and id(obj) != id(current_object):
                            current_object = obj
                            display_settings = DisplaySettings(len(current_object.data['label_name']), obj.data_type)

        imgui.end_child()

        imgui.same_line()

        imgui.begin_child("right pane", border=True, flags=imgui.WINDOW_NO_SCROLLBAR)
        imgui.text("Image")
        if not cached_output_image is None:
            size = Vec2(imgui.get_window_content_region_max()) - (Vec2(imgui.get_window_content_region_min()) + Vec2(imgui.get_cursor_position()))
            mouse_position = imgui_cv.image(cached_output_image, width=size.x, height=size.y)

        imgui.end_child()  
        imgui.end()


        gl.glClearColor(1.0, 1.0, 1.0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

def initialize_window():
    rich.print("[yellow][WARNING] [/yellow]initialize_window 将启动一个线程，这可能显著拖慢速度")
    global initialized
    if not initialized:
        thread = threading.Thread(target=main_loop)
        thread.start()
        while not initialized:
            time.sleep(0.01)

default_label_names = ['background', 'spleen', 'right kidney', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta', 'inferior vena cava', 'portal vein and splenic vein', 'pancreas', 'right adrenal gland', 'left adrenal gland']

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise Exception("Unknown type")

def add_object(name, data_type, image, pd_label=None, gt_label=None, prompt_points=None, label_name=None, background_label=0):
    """
    添加一个对象到可视化窗口中

    Args:
        name (str): 对象名称
        data_type (str): 数据类型，可以是 "2D" 或 "3D"
        image (torch.Tensor or np.ndarray): 二维或三维图像. Channel 应该在第一维.
        pd_label (torch.Tensor or np.ndarray, optional): 预测标签. 默认为 None.
        gt_label (torch.Tensor or np.ndarray, optional): 真实标签. 默认为 None.
        prompt_points (list, optional): prompt 点的列表. 对于三维数据，该列表是每个切片的二维 Prompt 的列表. 对于一个二维切片，其 prompt 列表中每一项的格式为 ((x, y), label). label 为 1 或 0. 默认为 None.
        label_name (list, optional): 标签名称列表. 使用原数据集的标签时，可以传入 visualize.default_label_names. 默认为 None.
        background_label (int, optional): 计算 Dice 时忽略的背景标签. 默认为 0.

    Example:
        2D 数据
        >>> import utils.visualize as visualize
        >>> import torch
        >>> visualize.initialize_window()
        >>> image = torch.rand(3, 256, 256)
        >>> pd_label = torch.randint(0, 14, (1, 256, 256))
        >>> gt_label = torch.randint(0, 14, (1, 256, 256))
        >>> prompt_points = [((100, 100), 1), ((200, 200), 0)]
        >>> visualize.add_object("test2D", "2D", 
        >>>                         image=image,
        >>>                         pd_label=pd_label,
        >>>                         gt_label=gt_label,
        >>>                         prompt_points=prompt_points,
        >>>                         label_name=visualize.default_label_names)

        3D 数据
        >>> import utils.visualize as visualize
        >>> import torch
        >>> visualize.initialize_window()
        >>> image = torch.rand(3, 256, 256, 256)
        >>> pd_label = torch.randint(0, 14, (1, 256, 256, 256))
        >>> gt_label = torch.randint(0, 14, (1, 256, 256, 256))
        >>> prompt_points = [[((100, 100), 1), ((200, 200), 0)], [((100, 100), 1), ((200, 200), 0)]] * 128
        >>> visualize.add_object("test3D", "3D",
        >>>                         image=image,
        >>>                         pd_label=pd_label,
        >>>                         gt_label=gt_label,
        >>>                         prompt_points=prompt_points,
        >>>                         label_name=visualize.default_label_names)
    """
    global objects
    image = to_numpy(image).copy()

    if data_type == "3D":
        image = np.moveaxis(image, 0, 1) # (C, D, H, W) -> (D, C, H, W)

    if pd_label is not None:
        pd_label = to_numpy(pd_label).copy()
        if data_type == "2D" and len(pd_label.shape) == 3:
            pd_label = pd_label[0]
        if data_type == "3D" and len(pd_label.shape) == 4:
            pd_label = pd_label[0]

    if gt_label is not None:
        gt_label = to_numpy(gt_label).copy()
        if data_type == "2D" and len(gt_label.shape) == 3:
            gt_label = gt_label[0]
        if data_type == "3D" and len(gt_label.shape) == 4:
            gt_label = gt_label[0]
        

    if prompt_points is not None:
        prompt_points = prompt_points.copy()

    if not pd_label is None:
        if pd_label.dtype != np.int32:
            pd_label = pd_label.astype(np.int32)
    
    if not gt_label is None:
        if gt_label.dtype != np.int32:
            gt_label = gt_label.astype(np.int32)

    max_label = max(gt_label.max() if not gt_label is None else 0, pd_label.max() if not pd_label is None else 0)
    if label_name is None:
        label_name = [str(i) for i in range(max_label + 1)]

    assert len(label_name) <= len(colors), "The number of labels must be less than %d" % len(colors)
    assert (len(label_name) >= max_label + 1), "The length of label_name should be greater than or equal to the number of labels in pd_label / gt_label"

    objects.append(Object(
        name, data_type,
        {
            'image': image,
            'pd_label': pd_label,
            'gt_label': gt_label,
            'prompt_points': prompt_points,
            'label_name': label_name,
            'background_label': background_label
        }
    ))

def add_object_2d(name, *, image, pd_label=None, gt_label=None, prompt_points=None, label_name=None, background_label=0):
    add_object(name, "2D", image, pd_label, gt_label, prompt_points, label_name, background_label)

def add_object_3d(name, *, image, pd_label=None, gt_label=None, prompt_points=None, label_name=None, background_label=0):
    add_object(name, "3D", image, pd_label, gt_label, prompt_points, label_name, background_label)

if __name__=="__main__":
    initialize_window()
    loader = dataset.get_data_loader("training", "naive_to_rgb", batch_size=1, shuffle=False, device="cpu", first_only=True)
    for data in loader:
        if (data['label'][0] != 0).sum() <= 1000:
            continue
        add_object_2d("test", 
                        image=data['image'][0],
                        pd_label=None,
                        gt_label=None,
                        prompt_points=[((100, 100), 1), ((200, 200), 0)],
                        label_name=default_label_names,
                        background_label=0)
        break