import sys
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, 
                             QMainWindow, 
                             QVBoxLayout, 
                             QHBoxLayout,
                             QWidget, 
                             QSlider, 
                             QPushButton, 
                             QLineEdit, 
                             QLabel,  
                             QGraphicsScene, 
                             QGraphicsPixmapItem, 
                             QGraphicsView, 
                             QMessageBox, 
                             QTextEdit, 
                             QFileDialog,
                             QSplitter)
from PyQt5.QtCore import Qt, QSize, QRect, QThread, pyqtSignal, pyqtSlot, QPointF, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor
import argparse
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import skimage
import math
from skimage.exposure import equalize_hist
from collections import defaultdict
from PIL import ImageColor
from scipy.ndimage import find_objects
from skimage.segmentation import watershed
from tqdm import tqdm
from PyQt5.QtGui import QMovie
from scipy.ndimage import label, find_objects, distance_transform_edt
#from postprocess import find_watershed_locations, filter_watershed_locations, refine_watershed

# created with https://github.com/lmcinnes/glasbey
# this package is sometimes stubborn to install, which is why it is hardcoded
glasbey_cmap_100 = ['#d21820', '#1869ff', '#008a00', '#f36dff', '#710079', '#aafb00', '#00bec2', 
                '#ffa235', '#5d3d04', '#08008a', '#005d5d', '#9a7d82', '#a2aeff', '#96b675', 
                '#9e28ff', '#4d0014', '#ffaebe', '#ce0092', '#00ffb6', '#002d00', '#9e7500', 
                '#3d3541', '#f3eb92', '#65618a', '#8a3d4d', '#5904ba', '#558a71', '#b2bec2', 
                '#ff5d82', '#1cc600', '#92f7ff', '#2d86a6', '#395d28', '#ebceff', '#ff5d00', 
                '#a661aa', '#860000', '#350059', '#00518e', '#9e4910', '#cebe00', '#002828', 
                '#00b2ff', '#caa686', '#be9ac2', '#2d200c', '#756545', '#8279df', '#00c28a', 
                '#bae7c2', '#868ea6', '#ca7159', '#829a00', '#2d00ff', '#d204f7', '#ffd7be', 
                '#92cef7', '#ba5d7d', '#ff41c2', '#be86ff', '#928e65', '#a604aa', '#86e375', 
                '#49003d', '#fbef0c', '#69555d', '#59312d', '#6935ff', '#b6044d', '#5d6d71', 
                '#414535', '#657100', '#790049', '#1c3151', '#79419e', '#ff9271', '#ffa6f3', 
                '#ba9e41', '#82aa9a', '#d77900', '#493d71', '#51a255', '#e782b6', '#d2e3fb', 
                '#004931', '#6ddbc2', '#3d4d5d', '#613555', '#007151', '#5d1800', '#9a5d51', 
                '#558edb', '#caca9a', '#351820', '#393d00', '#009a96', '#eb106d', '#8a4579', 
                '#75aac2', '#ca929a']

glasbey_map_1000 = ['#d21820', '#1869ff', '#008a00', '#f36dff', '#710079', '#aafb00', '#00bec2', '#ffa235', 
                    '#5d3d04', '#08008a', '#005d5d', '#9a7d82', '#a2aeff', '#96b675', '#9e28ff', '#4d0014', 
                    '#ffaebe', '#ce0092', '#00ffb6', '#002d00', '#9e7500', '#3d3541', '#f3eb92', '#65618a', 
                    '#8a3d4d', '#5904ba', '#558a71', '#b2bec2', '#ff5d82', '#1cc600', '#92f7ff', '#2d86a6', 
                    '#395d28', '#ebceff', '#ff5d00', '#a661aa', '#860000', '#350059', '#00518e', '#9e4910', 
                    '#cebe00', '#002828', '#00b2ff', '#caa686', '#be9ac2', '#2d200c', '#756545', '#8279df', 
                    '#00c28a', '#bae7c2', '#868ea6', '#ca7159', '#829a00', '#2d00ff', '#d204f7', '#ffd7be', 
                    '#92cef7', '#ba5d7d', '#ff41c2', '#be86ff', '#928e65', '#a604aa', '#86e375', '#49003d', 
                    '#fbef0c', '#69555d', '#59312d', '#6935ff', '#b6044d', '#5d6d71', '#414535', '#657100', 
                    '#790049', '#1c3151', '#79419e', '#ff9271', '#ffa6f3', '#ba9e41', '#82aa9a', '#d77900', 
                    '#493d71', '#51a255', '#e782b6', '#d2e3fb', '#004931', '#6ddbc2', '#3d4d5d', '#613555', 
                    '#007151', '#5d1800', '#9a5d51', '#558edb', '#caca9a', '#351820', '#393d00', '#009a96', 
                    '#eb106d', '#8a4579', '#75aac2', '#ca929a', '#d2bac6', '#9ace00', '#456daa', '#755900', 
                    '#ce4d0c', '#00dffb', '#ff3d41', '#ffca49', '#2d3192', '#866986', '#9e82be', '#ceaeff', 
                    '#79452d', '#c6fb82', '#5d7549', '#b64549', '#ffdfef', '#a20071', '#4d4da6', '#a6aaca', 
                    '#711c28', '#287979', '#084900', '#006986', '#a67549', '#fbb682', '#55187d', '#00ff59', 
                    '#00414d', '#6d8e92', '#aa2400', '#bed26d', '#8a61ba', '#d241be', '#496151', '#cef3ef', 
                    '#61c261', '#148a4d', '#00ffe7', '#006900', '#b2799e', '#aab29e', '#ba55ff', '#c679ce', 
                    '#203120', '#7d04db', '#c2c6f7', '#8ac6ce', '#e7ebce', '#281c39', '#9effae', '#82ce9a', 
                    '#31a60c', '#00a275', '#db9255', '#3d1404', '#ff8a9a', '#828635', '#694d71', '#b66100', 
                    '#7d2d00', '#a2b239', '#31047d', '#a63dca', '#9a202d', '#04df86', '#757d6d', '#8a96d2', 
                    '#08a2ca', '#f76d5d', '#1055ca', '#dbb665', '#92596d', '#a2ffe3', '#595528', '#7179aa', 
                    '#d75965', '#492051', '#df4d92', '#0000ca', '#5d65d2', '#dfa600', '#b24992', '#b68a75', 
                    '#614d3d', '#a696a2', '#551c35', '#314141', '#757586', '#929ea2', '#759a71', '#ff8220', 
                    '#8655ff', '#9ac6b6', '#df96f3', '#cadf31', '#8e5d28', '#35bee3', '#71a6ff', '#598a31', 
                    '#ffc2eb', '#aa3d69', '#49617d', '#49351c', '#45b29e', '#1c2431', '#f731ef', '#7500a6', 
                    '#e7b6aa', '#826965', '#e3a2ca', '#202400', '#79b610', '#9e8eff', '#d2758a', '#cab6db', 
                    '#ae9adf', '#ff71db', '#d2f7b2', '#c6d7ce', '#ffd28a', '#5ddf35', '#5d7992', '#a28e00', 
                    '#aedfef', '#714dc2', '#7d4500', '#6592b6', '#5d79ff', '#514959', '#969e51', '#ce69ae', 
                    '#653575', '#dbd2e3', '#b6ae75', '#515900', '#b65939', '#5504eb', '#3d752d', '#92829a', 
                    '#822469', '#ba8639', '#8ab2e3', '#6db282', '#964135', '#6d4149', '#8a753d', '#b27175', 
                    '#921c49', '#df6d31', '#00e3df', '#9204ca', '#312859', '#007dd2', '#a26dff', '#825992', 
                    '#a6dfd7', '#8e9a8a', '#3d4d79', '#d21849', '#aadb96', '#314924', '#8aebba', '#5d3d9a', 
                    '#00516d', '#5da6aa', '#651855', '#db8e79', '#862d8e', '#653114', '#5d5d69', '#d759fb', 
                    '#690008', '#a6bed7', '#ff65aa', '#003975', '#db4939', '#8e7961', '#3d5555', '#ca416d', 
                    '#008a75', '#752d4d', '#085d39', '#6d692d', '#96ca59', '#141c55', '#7daa49', '#8671a2', 
                    '#b29665', '#d7c6b2', '#ebdf55', '#9a459e', '#92ff71', '#c2a6a6', '#c208ae', '#4df78e', 
                    '#4d2400', '#412d2d', '#4d7569', '#a296b6', '#4149ff', '#b60024', '#a6c69e', '#351435', 
                    '#f7c6ce', '#be55ce', '#aa6dce', '#554145', '#ce8e00', '#00dba6', '#002d3d', '#7d554d', 
                    '#ef0ca2', '#798ef7', '#0041ae', '#3d656d', '#490000', '#6d65ae', '#3d2d00', '#c2ba51', 
                    '#00b649', '#aa9e8a', '#ebb6ff', '#20cab6', '#d7fb45', '#793531', '#412882', '#5d9a8e', 
                    '#715124', '#65002d', '#393d59', '#df8ad2', '#9e5d8a', '#ef410c', '#a2e749', '#ffaa96', 
                    '#ff358e', '#b6aebe', '#be3d24', '#51395d', '#598a55', '#0c4945', '#bea600', '#dfce71', 
                    '#551c18', '#45283d', '#ff96be', '#9e8a41', '#00659e', '#92b2b6', '#61a2db', '#4196a6', 
                    '#ce006d', '#794d65', '#61c2ff', '#c2753d', '#f33d61', '#450024', '#fbe7b2', '#5d5d49', 
                    '#5d4d82', '#18d759', '#009a39', '#df61d2', '#827100', '#6dc2a6', '#3d0049', '#9aa67d', 
                    '#006d35', '#75964d', '#2d5545', '#5d79c6', '#758659', '#554900', '#550065', '#1c86ff', 
                    '#103d18', '#3900ce', '#a22d8a', '#ba14ff', '#8a45c6', '#7ddfeb', '#ffba5d', '#0092d2', 
                    '#be04ce', '#a2c2ff', '#c68eae', '#9e6d82', '#8620ff', '#860024', '#ffb204', '#313514', 
                    '#00fbff', '#355100', '#dfbe8e', '#cedf9a', '#00392d', '#ff7541', '#249eff', '#ff8af7', 
                    '#e79e9e', '#2d202d', '#9e4d55', '#758600', '#2d3945', '#494118', '#9e7165', '#616d61', 
                    '#ff9e5d', '#416d49', '#9e75a6', '#7d868e', '#b65d5d', '#8e3561', '#6dbab6', '#c2cadf', 
                    '#e77d7d', '#697d79', '#8ea2ba', '#be8edb', '#4d828e', '#6531ce', '#b6d2f7', '#db9e41', 
                    '#d7a6df', '#f3e3ff', '#39599a', '#792014', '#716571', '#51516d', '#eb8e0c', '#d77dff', 
                    '#5d9608', '#2d695d', '#efca04', '#71559e', '#4982b6', '#455539', '#920086', '#002d14', 
                    '#922814', '#28009e', '#e7a279', '#416500', '#9675d2', '#00b269', '#d7a6b6', '#9e650c', 
                    '#3941c2', '#b6ca39', '#2defc2', '#8a86b6', '#3d2d4d', '#a23549', '#9e55df', '#fb92d7', 
                    '#556928', '#00aaba', '#55cadb', '#2d4535', '#82efe3', '#e392aa', '#4100aa', '#7dbedb', 
                    '#e78259', '#758e7d', '#450082', '#b286ba', '#c6dbe3', '#411018', '#e3ebf3', '#5d003d', 
                    '#31396d', '#04591c', '#8e6549', '#eb5d39', '#412061', '#dbd7fb', '#184161', '#497d00', 
                    '#412414', '#bec6b2', '#b2aee7', '#75f32d', '#2875a2', '#b6c27d', '#0469c2', '#b645b2', 
                    '#e7283d', '#794582', '#925131', '#ba7500', '#ce419e', '#b29292', '#d73100', '#a66131', 
                    '#4935aa', '#498a86', '#75e396', '#ef18c6', '#8eca7d', '#9e8a79', '#61a282', '#cadbba', 
                    '#dfb6d7', '#d7e77d', '#417186', '#5d6d96', '#ef7596', '#008a92', '#652d39', '#203900', 
                    '#757551', '#867582', '#8aba9a', '#ca7d79', '#b61871', '#968ad7', '#aefbca', '#614124', 
                    '#9edbbe', '#ae6951', '#9e040c', '#d7f3db', '#6dca41', '#397d59', '#c2b696', '#718ac2', 
                    '#2d2d45', '#aaef8e', '#203131', '#e3ba41', '#b2828e', '#db699a', '#b66596', '#555d9a', 
                    '#61697d', '#caa265', '#285561', '#aa3128', '#691cb2', '#aa8aa6', '#c6865d', '#9251ae', 
                    '#8265d2', '#614559', '#a2aeba', '#1c2871', '#45511c', '#354596', '#65ceca', '#eb69ba', 
                    '#aaced2', '#59d2ff', '#4d5d69', '#353120', '#55ca82', '#7da2a2', '#5151ce', '#6d61ff', 
                    '#b64d00', '#753969', '#5dae35', '#6d9aaa', '#962dae', '#aaae59', '#ffaadb', '#be9eb2', 
                    '#ce6104', '#9eb2aa', '#511c45', '#5db6c6', '#75a665', '#556541', '#c2f3ff', '#ffc2a6', 
                    '#697d39', '#d7c2ff', '#9a9e28', '#9e4979', '#b28610', '#db41e3', '#ebaa61', '#00822d', 
                    '#6d5d79', '#35eb65', '#b6a6ce', '#281059', '#8e3d18', '#41142d', '#d2414d', '#efceeb', 
                    '#e7d2aa', '#be5db2', '#b6a6ff', '#009e59', '#2d597d', '#82616d', '#ca5d4d', '#862839', 
                    '#e7efb6', '#65318e', '#5d7961', '#be71ef', '#391c00', '#8275ba', '#826524', '#5d2461', 
                    '#a60059', '#be4982', '#002445', '#00aeaa', '#925100', '#499669', '#966569', '#a68661', 
                    '#f35961', '#9e9eae', '#8e9eff', '#75ffce', '#ba3555', '#c67db6', '#202410', '#6d8a9e', 
                    '#e35d82', '#65b2df', '#ca9a8e', '#7d1039', '#babadb', '#4561ba', '#55313d', '#452020', 
                    '#86dfff', '#31201c', '#00798a', '#008e65', '#86597d', '#28a28e', '#715d51', '#e7f365', 
                    '#ce9aff', '#143dff', '#9669ae', '#ffdf6d', '#1c2041', '#efd2ce', '#51316d', '#d2ce49', 
                    '#e38239', '#b24d65', '#baefdb', '#824d55', '#654139', '#eb7100', '#002892', '#6d4900', 
                    '#b6008a', '#a6b6e3', '#a6752d', '#9e0039', '#d700d7', '#bee765', '#b2efae', '#aa82df', 
                    '#be0c04', '#86aa86', '#e34db2', '#a600e7', '#6d6500', '#be2d39', '#db2886', '#552492', 
                    '#b2f7f3', '#657d86', '#8e0c59', '#8a00aa', '#ba718a', '#0cb28a', '#ff757d', '#869661', 
                    '#a6492d', '#007969', '#496561', '#9eba00', '#651c7d', '#756d8e', '#006971', '#9aa6df', 
                    '#7d5539', '#351849', '#a659c2', '#ffbab6', '#6d55df', '#ff51df', '#df6961', '#4171d7', 
                    '#8e8204', '#ce7da2', '#6d3904', '#920ce7', '#51452d', '#51394d', '#7d9eca', '#ceb2a2', 
                    '#e35904', '#a6db75', '#969abe', '#144d24', '#8a4de3', '#4d4986', '#b2ceba', '#4d3931', 
                    '#692820', '#10c269', '#926596', '#610018', '#ae3d0c', '#a6be61', '#713db2', '#829692', 
                    '#1c310c', '#c66971', '#65d775', '#b6ff5d', '#b67969', '#86ba61', '#827d9a', '#b271ae', 
                    '#5992ff', '#790069', '#1ca2e3', '#864941', '#5db265', '#41414d', '#590055', '#e7db00', 
                    '#a65d6d', '#591424', '#003939', '#712871', '#ca6135', '#75b6ff', '#452df7', '#ffbeff', 
                    '#ff9a96', '#694582', '#fb8a4d', '#3d4d45', '#ff79c2', '#6582aa', '#ebb6ca', '#490469', 
                    '#24457d', '#419249', '#e7db96', '#515dff', '#005d4d', '#d28a31', '#ba9608', '#8a8a75', 
                    '#c2a2e3', '#db00b2', '#9e7d92', '#1ceba6', '#1c7900', '#71dbae', '#691c3d', '#8eaa10', 
                    '#9acee3', '#a69e61', '#659e3d', '#49415d', '#8296a2', '#4d7579', '#aeba8e', '#6151a2', 
                    '#698a69', '#92b2c6', '#6d5d2d', '#db82e7', '#102820', '#615910', '#c6bece', '#ba49e7', 
                    '#75f7ae', '#bad29e', '#75be79', '#5d491c', '#75288e', '#8ecec6', '#18e7ce', '#ef4dff', 
                    '#e34179', '#6d5586', '#9e55ff', '#928e9e', '#9a4965', '#55311c', '#ceaa3d', '#51a2ba', 
                    '#41c241', '#713d59', '#4d515d', '#8a6dff', '#f749a6', '#9e92c6', '#ca5996', '#efdfca', 
                    '#3d92ba', '#717d96', '#28454d', '#8e6904', '#79351c', '#49a671', '#c6966d', '#f3923d', 
                    '#399679', '#b296ca', '#ce65db', '#b6e700', '#d7e7e3', '#efae3d', '#759adf', '#7965a2', 
                    '#82922d', '#8a82ff', '#c2baff', '#590c0c', '#00285d', '#5d6dae', '#9edbaa', '#315535', 
                    '#dbc6e7', '#20007d', '#7171c6', '#6d82df', '#351810', '#cabe7d', '#79eb59', '#ebaee3', 
                    '#8ad745', '#8aebce', '#e3e3f7', '#ae203d', '#51b600', '#9e9ae7', '#65c292', '#cea6c6', 
                    '#9e4d8e', '#ff4d2d', '#968251', '#eb75df', '#9e6d92', '#71a604', '#59456d', '#b23582', 
                    '#6db296', '#aa00ca', '#e7c67d', '#c2ae5d', '#496d92', '#615565', '#18353d', '#fb82ae', 
                    '#8adf00', '#ef9e04', '#39a63d', '#96aa96', '#0059f3', '#8286ca', '#4d00ca', '#ce92d7', 
                    '#b27dca', '#e7cedb', '#a65500', '#497d45', '#00caa2', '#6d4d4d', '#491000', '#14394d', 
                    '#79793d', '#31654d', '#285d08', '#4d3104', '#4d5982', '#fb8271', '#f3a2ff', '#359614', 
                    '#284910', '#8aba41', '#86ff92', '#716d59', '#82550c', '#a2e7eb', '#b6b614', '#a22d6d', 
                    '#86a6b2', '#bee38a', '#7d1cba', '#eb3528', '#dfb292', '#df6d4d', '#d296c6', '#927d28', 
                    '#513dc6', '#ffdf28', '#8abab2', '#969235', '#207945', '#4d009e', '#6159ba', '#a6beaa', 
                    '#8ea669', '#eb6979', '#fbca9a', '#ef9a86', '#0059aa', '#dfeb00', '#3d3986', '#9a3100', 
                    '#dfbaeb', '#dfc2c2', '#ba652d', '#966d39', '#b2cee3', '#8e317d', '#8a2d28', '#755971', 
                    '#829275', '#df315d', '#c6e7b2', '#0082ba', '#aa5149', '#5dce00', '#2d4d69', '#6d1400', 
                    '#516979', '#413161', '#2d3100', '#4d5141', '#003d04', '#aa82fb', '#2daedf', '#51200c', 
                    '#c6556d', '#00c6ff', '#8292ba', '#0024d7', '#751c55', '#005d71', '#59969a', '#655535', 
                    '#928aaa', '#410c08', '#557924', '#bee3ff', '#dfa696', '#db8a8e', '#b6a682', '#c69649', 
                    '#ae31a2', '#612845', '#2d3d24', '#b6baca', '#aa35b6', '#7d658e', '#6daea2', '#efa2be', 
                    '#658a82', '#55242d', '#d775c6', '#283931', '#eb9ae3', '#512d4d', '#61efff', '#8aaaef', 
                    '#a669e7', '#caf7ca', '#b669c2', '#ebc6aa', '#ebaeb2', '#7504fb', '#8e1418', '#395d71', 
                    '#515d28', '#49353d', '#5586a2', '#82048e', '#613d65', '#be316d', '#312086', '#f79aaa', 
                    '#be4d3d', '#593d7d', '#69fff7', '#dfdfb2', '#00d2d7', '#71dfdb', '#ffd2ff', '#a29a75']

glasbey_cmap_20 = ['#d21820',
                '#1869ff',
                '#008a00',
                '#f36dff',
                '#710079',
                '#aafb00',
                '#00bec2',
                '#ffa235',
                '#5d3d04',
                '#08008a',
                '#005d5d',
                '#9a7d82',
                '#a2aeff',
                '#96b675',
                '#9e28ff',
                '#4d0014',
                '#ffaebe',
                '#ce0092',
                '#00ffb6',
                '#002d00']

# tab10_colormap = matplotlib.colormaps['tab10']
# glasbey_cmap = [mcolors.to_hex(color) for color in tab10_colormap.colors]
# del glasbey_cmap[7] # get rid of the grey one
glasbey_cmap = glasbey_map_1000
glasbey_cmap_rgb = [ImageColor.getcolor(col, "RGB") for col in glasbey_cmap]
num_colors = len(glasbey_cmap)

# icon from https://icons8.com/icon/52955/paint

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFormLayout, QSpinBox, QSlider, QDialog

class LoadingScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up the loading screen
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(256, 256)

        # Create a layout and add a label to hold the animation
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        layout.setAlignment(Qt.AlignCenter)  # Center the layout

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)  # Center the QLabel within the layout
        layout.addWidget(self.label)

        # Load the spinning wheel GIF with transparency
        self.movie = QMovie("loading.gif")
        self.label.setMovie(self.movie)
        self.movie.start()

        self.setLayout(layout)

        # Center the loading screen in the parent window
        self.center_on_parent()

    def center_on_parent(self):
        if self.parent():
            # Calculate the center position relative to the parent window
            parent_geometry = self.parent().geometry()
            parent_center = parent_geometry.center()
            self.move(parent_center.x() - self.width() // 2,
                      parent_center.y() - self.height() // 2)
        else:
            # No parent, center on the screen
            screen_geometry = QApplication.desktop().screenGeometry()
            screen_center = screen_geometry.center()
            self.move(screen_center.x() - self.width() // 2,
                      screen_center.y() - self.height() // 2)
            

class WatershedDialog(QDialog):
    def __init__(self, image_block, min_shift, threshold, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Watershed Parameters")
        
        self.initial_background_threshold = threshold
        self.current_threshold = threshold
        self.min_shift = min_shift
        self.image_block = image_block
        
        self.layout = QVBoxLayout()
        
        self.slider_label = QLabel(f"Threshold: {self.initial_background_threshold}", self)
        self.layout.addWidget(self.slider_label)
        
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 255)  
        self.slider.setValue(self.initial_background_threshold)
        self.slider.valueChanged.connect(self.update_preview)
        self.layout.addWidget(self.slider)
        
        self.preview_label = QLabel(self)
        self.layout.addWidget(self.preview_label)
        
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)
        
        self.setLayout(self.layout)
        
        # Call initial preview update
        self.update_preview()

    def update_preview(self):
        self.current_threshold = self.slider.value()
        self.slider_label.setText(f"Threshold: {self.current_threshold}")
        if self.parent:
            self.parent().watershed_background_threshold = self.current_threshold
            mask = self.image_block > self.current_threshold
            keep_points = np.argwhere(mask) #  > self.current_threshold
            block_shape = self.image_block.shape
            #keep_points = np.array([[block_shape[0]-1, 0, block_shape[2]//2]])
            #print("keep points", keep_points)
            #print("intensities", self.image_block[keep_points[:, 0], keep_points[:, 1], keep_points[:, 2]])
            #keep_points[:, 1] = block_shape[1] - keep_points[:, 1] 
            #keep_points[:, 2] = block_shape[2] - keep_points[:, 2] - 1
            #keep_points[:, 0] = block_shape[0] - keep_points[:, 0] 
            keep_points += self.min_shift
            #keep_points = keep_points[keep_points[:, 2], keep_points[:, 1], keep_points[:, 0]]
            keep_points = keep_points[:, [2, 1, 0]]
            #print("keep points after shift", keep_points)
            self.parent().watershed_foreground_points = keep_points
            self.parent().update_xy_view()
            self.parent().update_xz_view()
            self.parent().update_yz_view()

            
class ThresholdDialog(QDialog):
    def __init__(self, image_min, image_max, background_threshold, cell_centre_threshold, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Set Thresholds")
        
        self.image_min = image_min
        self.image_max = image_max
        self.background_threshold = background_threshold
        self.cell_centre_threshold = cell_centre_threshold
        
        # Create layout
        layout = QFormLayout()

        # Create sliders and spin boxes
        self.background_slider = QSlider(Qt.Horizontal, self)
        self.background_slider.setRange(image_min, image_max)
        self.background_slider.setValue(background_threshold)
        
        self.background_spinbox = QSpinBox(self)
        self.background_spinbox.setRange(image_min, image_max)
        self.background_spinbox.setValue(background_threshold)
        
        self.cell_centre_slider = QSlider(Qt.Horizontal, self)
        self.cell_centre_slider.setRange(image_min, image_max)
        self.cell_centre_slider.setValue(cell_centre_threshold)
        
        self.cell_centre_spinbox = QSpinBox(self)
        self.cell_centre_spinbox.setRange(image_min, image_max)
        self.cell_centre_spinbox.setValue(cell_centre_threshold)
        
        # Add widgets to layout
        layout.addRow(QLabel("Background Threshold:"), self.background_slider)
        layout.addRow("", self.background_spinbox)
        layout.addRow(QLabel("Cell Centre Threshold:"), self.cell_centre_slider)
        layout.addRow("", self.cell_centre_spinbox)
        
        # Create and add buttons
        button_layout = QVBoxLayout()
        self.ok_button = QPushButton("OK", self)
        self.cancel_button = QPushButton("Cancel", self)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addRow(button_layout)
        
        self.setLayout(layout)
        
        # Connect signals
        self.background_slider.valueChanged.connect(self.background_spinbox.setValue)
        self.background_spinbox.valueChanged.connect(self.background_slider.setValue)
        self.cell_centre_slider.valueChanged.connect(self.cell_centre_spinbox.setValue)
        self.cell_centre_spinbox.valueChanged.connect(self.cell_centre_slider.setValue)
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def get_values(self):
        # Return the adjusted values
        return (self.background_slider.value(), self.cell_centre_slider.value())

class MaskLoader(QThread):
    finished = pyqtSignal()  # Signal to indicate the task is finished
    progress = pyqtSignal(int)  # Signal to indicate progress (optional)

    def __init__(self, parent, filename, load_background=False):
        super().__init__()
        self.parent = parent
        self.filename = filename
        self.load_background = load_background

    def run(self):
        assert self.filename.endswith(".npy"), "Mask file must end with .npy"
        mask = np.load(self.filename)
        assert mask.shape == self.parent.image_data.shape, "Mask shape does not match image shape"
        self.parent.background_points = []
        self.parent.foreground_points = []
        print("loading masks...")
        slices = find_objects(mask)
        for i, slice_tuple in enumerate(tqdm(slices), start=1):
            if slice_tuple is not None:
                local_locs = np.array(np.where(mask[slice_tuple] == i))
                global_locs = np.stack(local_locs).T + np.array([s.start for s in slice_tuple])
                color_idx = i % num_colors
                for loc in global_locs:
                    z, y, x = loc
                    self.parent.foreground_points.append((x, y, z, i, color_idx))
                if i > self.parent.index_control.cell_index:
                    self.parent.index_control.cell_index = i
            # Emit progress signal (optional, depending on the implementation)
            self.progress.emit(i)
        
        # Optionally process background points
        if self.load_background:
            background_locs = np.array(np.where(mask == 0)).T
            self.parent.background_points = [tuple(loc) for loc in background_locs]

        # sort according to color index
        self.parent.foreground_points = sorted(self.parent.foreground_points, key=lambda x: x[-1])
        self.parent.current_highest_cell_index = self.parent.index_control.cell_index

        # Emit the finished signal
        self.finished.emit()

class Estimating_Cell_Thresholds(QThread):
    finished = pyqtSignal()  # Signal to indicate the task is finished
    progress = pyqtSignal(int)  # Signal to indicate progress (optional)
    cell_thresholds = pyqtSignal(tuple)

    def __init__(self, image, points):
        super().__init__()
        self.image = image
        self.points = points

    def minimum_of_brightest_spots(self, image, points):
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        z_coords = points[:, 2]
        cell_indices = points[:, 3]
        unique_cell_indices = np.unique(cell_indices)
        maximum_brightness_per_cell = []
        for cell_idx in unique_cell_indices:
            mask = (cell_indices == cell_idx)
            filtered_x = x_coords[mask]
            filtered_y = y_coords[mask]
            filtered_z = z_coords[mask]
            pixel_values = image[filtered_x, filtered_y, filtered_z]
            max_brightness = np.max(pixel_values)
            maximum_brightness_per_cell.append(max_brightness)

        maximum_brightness_per_cell = np.array(maximum_brightness_per_cell)
        minimum_median_brightness = np.percentile(maximum_brightness_per_cell, 25)
        return minimum_median_brightness

    def run(self):
        print("Estimating thresholds...")
        foreground = [point[:4] for point in self.points]
        foreground = np.array(foreground)
        background_mask = np.ones_like(self.image, dtype=bool)
        background_mask[foreground[:, 0], foreground[:, 1], foreground[:, 2]] = False
        minimum_centre_brightness = self.minimum_of_brightest_spots(self.image, foreground)
        #minimum_centre_brightness *= 0.8
        median_background_value = np.median(background_mask * self.image)
        self.cell_thresholds.emit((minimum_centre_brightness, median_background_value))
        self.finished.emit()

class IndexControlWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.cell_index = 1
        layout = QHBoxLayout()
        self.decrease_button = QPushButton('<')
        self.decrease_button.clicked.connect(self.decrease_index)
        layout.addWidget(self.decrease_button)
        self.index_label = QLabel(str(self.cell_index))
        layout.addWidget(self.index_label)
        self.increase_button = QPushButton('>')
        self.increase_button.clicked.connect(self.increase_index)
        layout.addWidget(self.increase_button)
        self.setLayout(layout)
        
    def decrease_index(self):
        if self.cell_index == 1:
            return
        self.cell_index -= 1
        self.index_label.setText(str(self.cell_index))
        self.repaint()
        
    def increase_index(self):
        self.cell_index += 1
        self.index_label.setText(str(self.cell_index))
        self.repaint()

    def update_index(self, number, highest_index):
        self.cell_index = number
        self.highest_index = highest_index
        self.index_label.setText(str(self.cell_index) + "/" + str(self.highest_index))
        self.repaint()

class TextDisplay(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)  
        self.setMaximumHeight(40)
        self.setMaximumWidth(170)
        self.setAlignment(Qt.AlignCenter)
        
    def update_text(self, number, highest_index):
        self.setText("Selected Cell Index: " + str(number) + " \nHighest Cell Index: " + str(highest_index))
        self.repaint()

class GraphicsView(QGraphicsView):
    def __init__(self, main_window, view_plane, parent=None):
        super().__init__(parent)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.main_window = main_window
        self.view_plane = view_plane
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pixmap_item)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        if self.view_plane == "XY":
            self.fixed_dim = "Z"
            self.missing_view_planes = ["XZ", "YZ"]
        elif self.view_plane == "XZ":
            self.fixed_dim = "Y"
            self.missing_view_planes = ["XY", "YZ"]
        elif self.view_plane == "YZ":
            self.fixed_dim = "X"
            self.missing_view_planes = ["XY", "XZ"]
        else:
            raise ValueError("Invalid viewplane.\
                             Choose among 'XY', 'XZ', 'YZ'")

    def setPixmap(self, pixmap):
        self._pixmap_item.setPixmap(pixmap)

    def rotate_view(self, angle):
        self.rotate(angle)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_1:
            current_value = self.main_window.slider.value()
            step_size = self.main_window.slider.singleStep()
            self.main_window.slider.setValue(current_value - step_size)
        elif event.key() == Qt.Key_2:
            current_value = self.main_window.slider.value()
            step_size = self.main_window.slider.singleStep()
            self.main_window.slider.setValue(current_value + step_size)
        elif event.key() == Qt.Key_3:
            current_value = self.main_window.slidery.value()
            step_size = self.main_window.slidery.singleStep()
            self.main_window.slidery.setValue(current_value - step_size)
        elif event.key() == Qt.Key_4:
            current_value = self.main_window.slidery.value()
            step_size = self.main_window.slidery.singleStep()
            self.main_window.slidery.setValue(current_value + step_size)
        elif event.key() == Qt.Key_5:
            current_value = self.main_window.sliderx.value()
            step_size = self.main_window.sliderx.singleStep()
            self.main_window.sliderx.setValue(current_value - step_size)
        elif event.key() == Qt.Key_6:
            current_value = self.main_window.sliderx.value()
            step_size = self.main_window.sliderx.singleStep()
            self.main_window.sliderx.setValue(current_value + step_size)
        elif event.key() == Qt.Key_M:
            if self.main_window.visualization_only:
                if self.main_window.markers_enabled:
                    if self.main_window.backup_greyscale is not None:
                        self.main_window.backup_color = self.main_window.image_data
                        self.main_window.image_data = self.main_window.backup_greyscale
                else:
                    if self.main_window.backup_color is not None:
                        self.main_window.backup_greyscale = self.main_window.image_data
                        self.main_window.image_data = self.main_window.backup_color
            self.main_window.markers_enabled = not self.main_window.markers_enabled
            self.main_window.update_xy_view()
            self.main_window.update_xz_view()
            self.main_window.update_yz_view()
        elif event.key() == Qt.Key_E:
            self.main_window.toggleEraser()
        elif event.key() == Qt.Key_J:
            self.main_window.visualization_mode()
        elif event.key() == Qt.Key_F:
            self.main_window.toggleForeground()
        elif event.key() == Qt.Key_B:
            self.main_window.toggleBackground()
        elif event.key() == Qt.Key_V:
            self.main_window.hide_show_view_finder()
        elif event.key() == Qt.Key_Left:
            self.main_window.index_control.decrease_index()
            self.main_window.update_index_display()
        elif event.key() == Qt.Key_Right:
            self.main_window.index_control.increase_index()
            self.main_window.update_index_display()
        elif event.key() == Qt.Key_C:
            self.main_window.findCell()
        elif event.key() == Qt.Key_S:
            self.main_window.select_cell()
        elif event.key() == Qt.Key_L:
            self.main_window.localContrastEnhancer()
        elif event.key() == Qt.Key_O:
            self.main_window.open_file_dialog()
        elif event.key() == Qt.Key_D:
            self.main_window.delete_cell()
        elif event.key() == Qt.Key_X:
            self.main_window.labelCorrections()
        elif event.key() == Qt.Key_Plus:
            if self.main_window.watershed_seeding_point and self.main_window.watershed_neighborhood_selection_enabled:
                self.main_window.watershed_radius += 2
                self.main_window.update_xy_view()
                self.main_window.update_xz_view()
                self.main_window.update_yz_view()
        elif event.key() == Qt.Key_Minus:
            if self.main_window.watershed_seeding_point and self.main_window.watershed_neighborhood_selection_enabled:
                if self.main_window.watershed_radius > 1:
                    self.main_window.watershed_radius -= 2
                    self.main_window.update_xy_view()
                    self.main_window.update_xz_view()
                    self.main_window.update_yz_view()
        elif event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            self.main_window.apply_watershed()
        event.accept()

    def obtain_current_points(self, pixmap_item, event, view_plane):
        if view_plane == "XY":
            sp = self.mapToScene(event.pos())
            lp = pixmap_item.mapFromScene(sp).toPoint()
            z_index = self.main_window.slider.value()
            return (lp.x(), lp.y(), z_index)
        elif view_plane == "XZ":
            sp = self.mapToScene(event.pos())
            lp = pixmap_item.mapFromScene(sp).toPoint()
            y_index = self.main_window.slidery.value()
            return (lp.y(), y_index, lp.x())
        elif view_plane == "YZ":
            sp = self.mapToScene(event.pos())
            lp = pixmap_item.mapFromScene(sp).toPoint()
            x_index = self.main_window.sliderx.value()
            return (x_index, lp.y(), lp.x())
        else:
            raise ValueError("Invalid viewplane.\
                             Choose among 'XY', 'XZ', 'YZ'")

    def wheelEvent(self, event, recursion = True):
        if recursion:
            self.main_window.synchronize_wheeling(self.missing_view_planes, event)
        factor = 1.2
        if event.angleDelta().y() < 0:
            factor = 0.8
        view_pos = event.pos()
        scene_pos = self.mapToScene(view_pos)
        self.centerOn(scene_pos)
        self.scale(factor, factor)
        delta = self.mapToScene(view_pos) - self.mapToScene(self.viewport().rect().center())
        self.centerOn(scene_pos - delta)
        event.accept()

    def generate_points_around(self, center_point, fixed_dimension, distance):
        points = []
        x, y, z = center_point
        print("center point", center_point)
        if fixed_dimension == 'X':
            for j in range(y - distance, y + distance + 1):
                for k in range(z - distance, z + distance + 1):
                    if (j - y) ** 2 + (k - z) ** 2 <= distance ** 2\
                    and j <= self.main_window.y_max \
                    and j >= self.main_window.y_min \
                    and k <= self.main_window.z_max \
                    and k >= self.main_window.z_min:
                        if self.main_window.foreground_enabled:
                            points.append((x, j, k,  self.main_window.index_control.cell_index, self.main_window.index_control.cell_index%num_colors))
                        else:
                            points.append((x, j, k))
        elif fixed_dimension == 'Y':
            for i in range(x - distance, x + distance + 1):
                for k in range(z - distance, z + distance + 1):
                    if (i - x) ** 2 + (k - z) ** 2 <= distance ** 2\
                    and i <= self.main_window.x_max \
                    and i >= self.main_window.x_min \
                    and k <= self.main_window.z_max \
                    and k >= self.main_window.z_min:
                        if self.main_window.foreground_enabled:
                            points.append((i, y, k,  self.main_window.index_control.cell_index, self.main_window.index_control.cell_index%num_colors))
                        else:
                            points.append((i, y, k))
        elif fixed_dimension == 'Z':
            for i in range(x - distance, x + distance + 1):
                for j in range(y - distance, y + distance + 1):
                    if (i - x) ** 2 + (j - y) ** 2 <= distance ** 2\
                    and i <= self.main_window.x_max \
                    and i >= self.main_window.x_min \
                    and j <= self.main_window.y_max \
                    and j >= self.main_window.y_min:
                        if self.main_window.foreground_enabled:
                            points.append((i, j, z, self.main_window.index_control.cell_index, self.main_window.index_control.cell_index%num_colors))
                        else:
                            points.append((i, j, z))
        return points

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:

            if self.main_window.delete_cell_enabled:
                pixmap_item = self._pixmap_item
                point = self.obtain_current_points(pixmap_item, event, self.view_plane)
                if point:
                    mbox = QMessageBox.question(self,  
                             "Delete Cell",  
                             "Are you sure you want to delete this cell?", 
                             QMessageBox.Yes | QMessageBox.No,  
                             QMessageBox.No)
                    if mbox == QMessageBox.Yes:
                        cell_idx = None
                        for p in self.main_window.foreground_points:
                            if tuple(p[:3]) == point:
                                cell_idx = p[3]
                                break
                        indices_to_remove = []
                        for i, p in enumerate(self.main_window.foreground_points):
                            if p[3] == cell_idx:
                                indices_to_remove.append(i)
                        self.main_window.remove_indices(indices_to_remove)
                event.accept()
                return
            
            if self.main_window.watershed_neighborhood_selection_enabled:
                pixmap_item = self._pixmap_item
                self.main_window.watershed_seeding_point = self.obtain_current_points(pixmap_item, event, self.view_plane)
                self.main_window.update_xy_view()
                self.main_window.update_xz_view()
                self.main_window.update_yz_view()


            if not self.main_window.new_cell_selected \
                and self.main_window.select_cell_enabled \
                and self.main_window.foreground_points \
                and self.main_window.markers_enabled:
                pixmap_item = self._pixmap_item
                point = self.obtain_current_points(pixmap_item, event, self.view_plane)
                if point:
                    cell_idx = None
                    for p in self.main_window.foreground_points:
                        if tuple(p[:3]) == point:
                            cell_idx = p[3]
                            break
                    if cell_idx:
                        self.main_window.new_cell_selected = True
                        self.main_window.index_control.cell_index = cell_idx
                        self.main_window.update_index_display()
                        self.main_window.index_control.update_index(self.main_window.index_control.cell_index, 
                                                                    self.main_window.current_highest_cell_index)
                        self.main_window.update_xy_view()
                        self.main_window.update_xz_view()
                        self.main_window.update_yz_view()
            else:
                self.main_window.dragging = True
                if self.main_window.local_contrast_enhancer_enabled:
                    pixmap_item = self._pixmap_item
                    sp = self.mapToScene(event.pos())
                    lp = pixmap_item.mapFromScene(sp).toPoint()
                    self.main_window.first_mouse_pos_for_contrast_rect = lp
                else:
                    if self.main_window.drawing and self.main_window.markers_enabled:
                        pixmap_item = self._pixmap_item
                        points = self.obtain_current_points(pixmap_item, event, self.view_plane)
                        #print("Point: ", points)
                        if self.main_window.foreground_enabled:
                            if self.main_window.brush_width == 1:
                                points = points + (self.main_window.index_control.cell_index,) + (self.main_window.index_control.cell_index%num_colors,)
                                if points not in self.main_window.foreground_points:
                                    self.main_window.foreground_points.append(points)
                            elif self.main_window.brush_width > 1:
                                ppoints = self.generate_points_around(points, self.fixed_dim, self.main_window.brush_width - 1)
                                for pp in ppoints:
                                    if pp not in self.main_window.foreground_points:
                                        self.main_window.foreground_points.append(pp)
                        elif self.main_window.background_enabled:
                            if self.main_window.brush_width == 1:
                                if points not in self.main_window.background_points:
                                    self.main_window.background_points.append(points)
                            elif self.main_window.brush_width > 1:
                                ppoints = self.generate_points_around(points, self.fixed_dim, self.main_window.brush_width - 1)
                                for pp in ppoints:
                                    if pp not in self.main_window.background_points:
                                        self.main_window.background_points.append(pp)
                        elif self.main_window.eraser_enabled:
                            self.main_window.removePoints(points, self.view_plane)
                        self.main_window.update_xy_view()
                        self.main_window.update_xz_view()
                        self.main_window.update_yz_view()
                    else:
                        self.main_window.last_mouse_pos = event.pos()

        elif event.button() == Qt.RightButton:

            if self.main_window.watershed_neighborhood_selection_enabled:
                self.main_window.watershed_seeding_point = None
                self.main_window.watershed_radius = 2
                self.main_window.update_xy_view()
                self.main_window.update_xz_view()
                self.main_window.update_yz_view()

            if self.main_window.copied_points:
                view_plane_index = 2 if self.view_plane == 'XY' else 1 if self.view_plane == 'XZ' else 0
                slider_value = self.main_window.slider.value() if view_plane_index == 2 else self.main_window.slidery.value() if view_plane_index == 1 else self.main_window.sliderx.value()
                for p in self.main_window.copied_points:
                    if p[-1] == view_plane_index:
                        new_point = list(p[:5])
                        new_point[view_plane_index] = slider_value
                        self.main_window.foreground_points.append(tuple(new_point))
                self.main_window.copied_points = []
                self.main_window.update_xy_view()
                self.main_window.update_xz_view()
                self.main_window.update_yz_view()
            else:
                pixmap_item = self._pixmap_item
                point = self.obtain_current_points(pixmap_item, event, self.view_plane)
                if point:
                    cell_idx = None
                    view_plane_index = 2 if self.view_plane == 'XY' else 1 if self.view_plane == 'XZ' else 0
                    slider_value = self.main_window.slider.value() if view_plane_index == 2 else self.main_window.slidery.value() if view_plane_index == 1 else self.main_window.sliderx.value()
                    for p in self.main_window.foreground_points:
                        if tuple(p[:3]) == point:
                            cell_idx = p[3]
                            break
                    if cell_idx:
                        for p in self.main_window.foreground_points:
                            if p[3] == cell_idx and p[view_plane_index] == slider_value:
                                self.main_window.copied_points.append(p + (p[-1],) + (view_plane_index,))
        event.accept()

    def mouseMoveEvent(self, event):

        if self.main_window.local_contrast_enhancer_enabled and self.main_window.dragging:
            pixmap_item = self._pixmap_item
            sp = self.mapToScene(event.pos())
            lp = pixmap_item.mapFromScene(sp).toPoint()
            self.main_window.last_mouse_pos_for_contrast_rect = lp
            self.main_window.update_xy_view()
            self.main_window.update_xz_view()
            self.main_window.update_yz_view()
        else:
            if self.main_window.drawing and self.main_window.dragging and self.main_window.markers_enabled:
                pixmap_item = self._pixmap_item
                points = self.obtain_current_points(pixmap_item, event, self.view_plane)
                if self.main_window.foreground_enabled:
                    if self.main_window.brush_width == 1:
                        points = points + (self.main_window.index_control.cell_index,) + (self.main_window.index_control.cell_index%num_colors,)
                        if points not in self.main_window.foreground_points:
                            self.main_window.foreground_points.append(points)
                    elif self.main_window.brush_width > 1:
                        ppoints = self.generate_points_around(points, self.fixed_dim, self.main_window.brush_width - 1)
                        for pp in ppoints:
                            if pp not in self.main_window.foreground_points:
                                self.main_window.foreground_points.append(pp)
                elif self.main_window.background_enabled:
                    if self.main_window.brush_width == 1:
                        if points not in self.main_window.background_points:
                            self.main_window.background_points.append(points)
                    elif self.main_window.brush_width > 1:
                        ppoints = self.generate_points_around(points, self.fixed_dim, self.main_window.brush_width - 1)
                        for pp in ppoints:
                            if pp not in self.main_window.background_points:
                                self.main_window.background_points.append(pp)
                elif self.main_window.eraser_enabled:
                    self.main_window.removePoints(points, self.view_plane)
                self.main_window.update_xy_view()
                self.main_window.update_xz_view()
                self.main_window.update_yz_view()
            elif self.main_window.dragging:
                delta = event.pos() - self.main_window.last_mouse_pos
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
                self.main_window.last_mouse_pos = event.pos()
        event.accept()

    def mouseReleaseEvent(self, event):
        self.main_window.dragging = False
        event.accept()

class MainWindow(QMainWindow):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        if filename[-3:] == "npy":
            self.image_data = np.load(filename)
        else:
            self.image_data = skimage.io.imread(filename)
        self.image_data = (self.image_data - self.image_data.min()) / (self.image_data.max() - self.image_data.min())
        self.image_data = (self.image_data * 255.0).astype(np.uint8)
        self.image_min = 0
        self.image_max = 255
        self.z_max = self.image_data.shape[0] - 1
        self.y_max = self.image_data.shape[1] - 1
        self.x_max = self.image_data.shape[2] - 1
        self.z_min = 0
        self.y_min = 0
        self.x_min = 0
        self.min_pixel_intensity = np.min(self.image_data.ravel())
        self.max_pixel_intensity = np.max(self.image_data.ravel())
        self.brush_width = 2
        self.eraser_radius = 2 
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        self.xy_view = GraphicsView(self, "XY")
        self.xz_view = GraphicsView(self, "XZ")
        self.yz_view = GraphicsView(self, "YZ")
        layout.addWidget(self.xy_view)
        layout.addWidget(self.xz_view)
        layout.addWidget(self.yz_view)

        layout_buttons = QHBoxLayout()
        layout_vertical_buttons = QVBoxLayout()

        self.slider = QSlider(Qt.Horizontal)
        self.slider_label = QLabel("Z-Planes (1,2)")
        self.slider.setRange(0, self.z_max)
        self.slider.setSingleStep(1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setValue(self.z_max//2)
        layout_buttons.addWidget(self.slider_label)
        layout_buttons.addWidget(self.slider)
        
        self.slidery = QSlider(Qt.Horizontal)
        self.slidery_label = QLabel("Y-Planes (3,4)")
        self.slidery.setRange(0, self.y_max)
        self.slidery.setSingleStep(1)
        self.slidery.setTickInterval(1)
        self.slidery.setTickPosition(QSlider.TicksBelow)
        self.slidery.setValue(self.y_max//2)
        layout_buttons.addWidget(self.slidery_label)
        layout_buttons.addWidget(self.slidery)
        
        self.sliderx = QSlider(Qt.Horizontal)
        self.sliderx_label = QLabel("X-Planes (5,6)")
        self.sliderx.setRange(0, self.x_max)
        self.sliderx.setSingleStep(1)
        self.sliderx.setTickInterval(1)
        self.sliderx.setTickPosition(QSlider.TicksBelow)
        self.sliderx.setValue(self.x_max//2)
        layout_buttons.addWidget(self.sliderx_label)
        layout_buttons.addWidget(self.sliderx)
        

        self.brush_label = QLabel("Brush Width:")
        layout_buttons.addWidget(self.brush_label)

        self.default_cursor = QCursor(Qt.ArrowCursor)
        self.cursor_pix = QPixmap('paintbrush_icon.png')
        self.cursor_scaled_pix = self.cursor_pix.scaled(QSize(50, 50), Qt.KeepAspectRatio)
        self.brush_cursor = self.default_cursor

        self.droplet_cursor_pix = QPixmap('droplet_cursor.png')
        self.droplet_cursor_scaled_pix = self.droplet_cursor_pix.scaled(QSize(20, 20), Qt.KeepAspectRatio)
        self.droplet_cursor = QCursor(self.droplet_cursor_scaled_pix)
        
        self.brush_text = QLineEdit()
        self.brush_text.setFixedWidth(30) 
        self.brush_text.setText(str(self.brush_width))
        self.brush_text.returnPressed.connect(self.updateBrushWidthFromLineEdit)
        layout_buttons.addWidget(self.brush_text)

        self.eraser_radius_label = QLabel("Eraser Radius:")
        layout_buttons.addWidget(self.eraser_radius_label)
        self.eraser_radius_text = QLineEdit()
        self.eraser_radius_text.setFixedWidth(30)
        self.eraser_radius_text.setText(str(self.eraser_radius))
        self.eraser_radius_text.returnPressed.connect(self.updateEraserRadius)
        layout_buttons.addWidget(self.eraser_radius_text)

        self.foreground_button = QPushButton('Foreground (F)', self)
        self.foreground_button.clicked.connect(self.toggleForeground)
        layout_buttons.addWidget(self.foreground_button)

        self.visualization_mode_button = QPushButton('Visualization Mode (J)', self)
        self.visualization_mode_button.clicked.connect(self.visualization_mode)
        layout_buttons.addWidget(self.visualization_mode_button)
        self.visualization_only = False
        self.backup_greyscale = None
        self.backup_color = None

        self.reset_transformations_button = QPushButton('Visualization Mode (J)', self)

        self.background_button = QPushButton('Background (B)', self)
        self.background_button.clicked.connect(self.toggleBackground)
        layout_buttons.addWidget(self.background_button)

        self.eraser_button = QPushButton('Eraser (E)', self)
        self.eraser_button.clicked.connect(self.toggleEraser)
        layout_buttons.addWidget(self.eraser_button)

        self.save_button = QPushButton('Save Masks', self)
        self.save_button.clicked.connect(self.saveMasks)
        layout_vertical_buttons.addWidget(self.save_button)

        self.find_cell_button = QPushButton('Find Cell (C)', self)
        self.find_cell_button.clicked.connect(self.findCell)
        layout_buttons.addWidget(self.find_cell_button)

        self.label_corrections_button = QPushButton('Label Corrections (X)', self)
        self.label_corrections_button.clicked.connect(self.labelCorrections)
        layout_buttons.addWidget(self.label_corrections_button)

        self.background_threshold = 0.0
        self.cell_centre_threshold = 0.0

        self.markers_off_on_button = QPushButton('Markers Off/On (M)', self)
        self.markers_off_on_button.clicked.connect(self.markersOffOn)
        layout_buttons.addWidget(self.markers_off_on_button)

        self.local_contrast_enhancer_button = QPushButton('Local Contrast Enhancer', self)
        self.local_contrast_enhancer_button.clicked.connect(self.localContrastEnhancer)
        self.local_contrast_enhancer_button.setText("Local Contrast Enhancement (L)")
        self.local_contrast_enhancer_enabled = False
        layout_buttons.addWidget(self.local_contrast_enhancer_button)

        self.submit_contrast_button = QPushButton('Submit Contrast', self)
        self.submit_contrast_button.clicked.connect(self.toggle_submit_contrast)
        self.submit_contrast_button.hide()
        layout_buttons.addWidget(self.submit_contrast_button)
    
        self.view_finder = True
        self.view_finder_button = QPushButton("Hide/Show View Finder (V)", self)
        self.view_finder_button.clicked.connect(self.hide_show_view_finder)
        layout_buttons.addWidget(self.view_finder_button)

        self.open_button = QPushButton('Open Mask (O)', self)
        self.open_button.clicked.connect(self.open_file_dialog)
        layout_vertical_buttons.addWidget(self.open_button)

        self.select_cell_button = QPushButton('Select Cell (S)', self)
        self.select_cell_button.clicked.connect(self.select_cell)
        self.select_cell_enabled = False
        self.new_cell_selected = False
        layout_vertical_buttons.addWidget(self.select_cell_button)

        self.delete_cell_button = QPushButton('Delete Cell (D)', self)
        self.delete_cell_button.clicked.connect(self.delete_cell)
        layout_vertical_buttons.addWidget(self.delete_cell_button)
        self.delete_cell_enabled = False

        self.index_control = IndexControlWidget()
        layout_vertical_buttons.addWidget(self.index_control)
        self.index_control.increase_button.clicked.connect(self.update_index_display)
        self.index_control.decrease_button.clicked.connect(self.update_index_display)
        self.current_highest_cell_index = 1

        self.cell_idx_display = TextDisplay()
        self.cell_idx_display.update_text(self.index_control.cell_index, self.current_highest_cell_index)
        layout_vertical_buttons.addWidget(self.cell_idx_display)
        layout.addLayout(layout_vertical_buttons)
        layout.addLayout(layout_buttons)

        self.progress_label = QLabel('Progress: 0%', self)
        self.progress_label.hide()
        layout.addWidget(self.progress_label)

        self.slider.valueChanged.connect(self.update_xy_view)
        self.slider.valueChanged.connect(self.update_yz_view)
        self.slider.valueChanged.connect(self.update_xz_view)
        self.slidery.valueChanged.connect(self.update_xz_view)
        self.sliderx.valueChanged.connect(self.update_yz_view)
        self.slidery.valueChanged.connect(self.update_xy_view)
        self.sliderx.valueChanged.connect(self.update_xy_view)
        self.slidery.valueChanged.connect(self.update_yz_view)
        self.sliderx.valueChanged.connect(self.update_xz_view)

        self.side_widget = QWidget()
        self.side_layout = QVBoxLayout(self.side_widget)
        self.hist_figure = Figure()
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_toolbar = NavigationToolbar(self.hist_canvas, self)
        self.plot_hist()
        self.side_layout.addWidget(self.hist_toolbar)
        self.side_layout.addWidget(self.hist_canvas)
        splitter = QSplitter()
        splitter.addWidget(self.central_widget)
        splitter.addWidget(self.side_widget)
        self.setCentralWidget(splitter)

        self.drawing = False  
        self.dragging = False
        self.temp_past_points = []
        self.xy_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  
        self.xy_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)    
        self.xz_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.xz_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.yz_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.yz_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.paint_color = QColor(Qt.red)
        self.paint_color.setAlphaF(0.3)
        self.foreground_enabled = False
        self.background_enabled = False
        self.watershed_neighborhood_selection_enabled = False
        self.watershed_seeding_point = None
        self.watershed_radius = 2
        self.watershed_foreground_points = []
        self.watershed_background_threshold = int(0.5 * 255)
        self.eraser_enabled = False
        self.last_mouse_pos = None
        self.first_mouse_pos_for_contrast_rect = None
        self.last_mouse_pos_for_contrast_rect = None
        self.first_mouse_pos_for_watershed_cube = None
        self.last_mouse_pos_for_watershed_cube = None
        self.yz_view.rotate_view(-90)
        self.xz_view.rotate_view(-90)
        self.foreground_points = []
        self.background_points = []
        self.copied_points = []
        self.markers_enabled = True
        self.update_xz_view()
        self.update_yz_view()
        self.update_xy_view()
        
    def numpyArrayToPixmap(self, img_np):
        img_np = np.require(img_np, np.uint8, 'C')
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            qim = QImage(img_np.data, img_np.shape[1], img_np.shape[0], img_np.strides[0], QImage.Format_RGB888)
        else:
            qim = QImage(img_np.data, img_np.shape[1], 
                         img_np.shape[0], img_np.strides[0], 
                         QImage.Format_Indexed8)
        pixmap = QPixmap.fromImage(qim)
        return pixmap
    
    def synchronize_wheeling(self, missing_view_planes, wheel_event):
        graphics_views = [self.xy_view, self.xz_view, self.yz_view]
        for view in graphics_views:
            if view.view_plane in missing_view_planes:
                view.wheelEvent(wheel_event, False)
    
    def visualization_mode(self):
        
        if not self.visualization_only:
            self.visualization_mode_button.setStyleSheet("background-color: lightgreen")
            self.repaint()
            img = self.image_data
            rgb_img = np.stack([img] * 3, axis=-1) 
            # sort according to color index
            points = sorted(self.foreground_points, key=lambda x: x[-1]) 
            self.visualization_only = True
            self.backup_greyscale = img
            batches = defaultdict(list)
            for point in points:
                batches[point[-1]].append(point[:3])
            for key in batches:
                points_list = batches[key]
                color_tuple = glasbey_cmap_rgb[key]
                for point in points_list:
                    x, y, z = point
                    rgb_img[z, y, x] = color_tuple
            self.image_data = rgb_img
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
        else:
            self.visualization_only = False
            self.visualization_mode_button.setStyleSheet("")
            self.repaint()
            self.backup_color = self.image_data
            self.image_data = self.backup_greyscale
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()

    def hide_show_view_finder(self):
        if self.view_finder:
            self.view_finder_button.setText("View Finder Off (V)")
        else:
            self.view_finder_button.setText("View Finder On (V)")
        self.view_finder = not self.view_finder
        self.update_xy_view()
        self.update_xz_view()
        self.update_yz_view()

    def select_cell(self):
        self.select_cell_enabled = not self.select_cell_enabled
        if self.select_cell_enabled:
            self.current_highest_cell_index = self.index_control.cell_index
            self.select_cell_button.setStyleSheet("background-color: lightgreen")
        else:
            self.select_cell_button.setStyleSheet("")
            self.new_cell_selected = False
            self.index_control.cell_index = self.current_highest_cell_index
            self.update_index_display()
            self.index_control.update_index(self.index_control.cell_index, self.current_highest_cell_index)
        self.repaint()

    def delete_cell(self):
        self.delete_cell_enabled = not self.delete_cell_enabled
        if self.delete_cell_enabled:
            self.delete_cell_button.setStyleSheet("background-color: lightgreen")
        else:
            self.delete_cell_button.setStyleSheet("")
        self.repaint()

    def apply_watershed(self):
        # this function is just garbage
        # it always gets the coordinates wrong, and I cannot understand why
        # even flipping coordinates doesn't work, so I'm leaving it here
        self.watershed_neighborhood_selection_enabled = not self.watershed_neighborhood_selection_enabled
        radius = self.watershed_radius
        seeding_point = self.watershed_seeding_point
        threshold = self.watershed_background_threshold
        self.watershed_seeding_point = None
        self.watershed_radius = 2
        image_block = self.image_data[seeding_point[2] - radius:seeding_point[2] + radius + 1,
                                      seeding_point[1] - radius:seeding_point[1] + radius + 1,
                                      seeding_point[0] - radius:seeding_point[0] + radius + 1]
        skimage.io.imsave("image_block.tif", image_block)
        min_shift = np.array([seeding_point[2] - radius, seeding_point[1] - radius, seeding_point[0] - radius])
        dialog = WatershedDialog(image_block, min_shift, threshold, self)
        if dialog.exec_() == QDialog.Accepted:
            print("hello")
        
        

    def labelCorrections(self):
        mbox = QMessageBox(self)
        if self.foreground_points:
            mbox.setWindowTitle("Correction Type")
            mbox.setText("Would you like to...?")

            button1 = mbox.addButton("Add cells missed by the labeler", QMessageBox.ActionRole)
            button2 = mbox.addButton("Correct already labeled cells", QMessageBox.ActionRole)
            button3 = mbox.addButton("Individual Cell Correction Through Watershed", QMessageBox.ActionRole)
            mbox.exec_()

            if mbox.clickedButton() == button1:
                mbox = QMessageBox(self)
                mbox.setWindowTitle("Labeling estimation")
                mbox.setText("Please enter your labeling setings in the following dialog. \n"
                            "You have to choose among minimum centre brightness and background threshold. "
                            "We have estimated these values based on the current image.\n")
                estimating_cell_thread = Estimating_Cell_Thresholds(self.image_data, self.foreground_points)
                estimating_cell_thread.progress.connect(self.handle_progress)
                estimating_cell_thread.cell_thresholds.connect(self.handle_thresholds)
                estimating_cell_thread.finished.connect(self.handle_finished)
                estimating_cell_thread.start()
                mbox.exec_()

                dialog = ThresholdDialog(self.image_min, self.image_max, self.background_threshold, self.cell_centre_threshold, self)
                if dialog.exec_() == QDialog.Accepted:
                    background_threshold, cell_centre_threshold = dialog.get_values()
                    # Update the class attributes or use the values as needed
                    self.background_threshold = background_threshold
                    self.cell_centre_threshold = cell_centre_threshold
                    foreground_points = [point[:3] for point in self.foreground_points]
                    foreground_points = np.array(foreground_points)
                    foreground_mask = np.zeros_like(self.image_data)
                    background_mask = np.ones_like(self.image_data)
                    foreground_mask[foreground_points[:, 0], foreground_points[:, 1], foreground_points[:, 2]] = 1
                    background_mask = (background_mask - foreground_mask).astype(int)
                    potential_new_cells = self.image_data > cell_centre_threshold
                    potential_new_cells[7:57, 7:57, 7:57] = False
                    labeled_array, num_features = label(potential_new_cells)
                    slices = find_objects(labeled_array)
                    seeding_points = []
                    for i, slice_tuple in enumerate(tqdm(slices)):
                        if slice_tuple is not None:
                            mom_locs = np.array(np.where(labeled_array[slice_tuple] == (i + 1)))
                            mom_locs_global = np.stack(mom_locs).T + np.array([s.start for s in slice_tuple])
                            brightest_within_loc = np.argmax(labeled_array[mom_locs_global[:, 0], mom_locs_global[:, 1], mom_locs_global[:, 2]])
                            brightest_coordinates = mom_locs_global[brightest_within_loc]
                            seeding_points.append(brightest_coordinates)
                            #print("brightest_coordinates", brightest_coordinates)
                    print("seeding_points", seeding_points)
                    seeding_points = np.array(seeding_points)
                    #print("seeding_points", seeding_points)
                    inverted_image = np.ones_like(self.image_data) * 255.0 - self.image_data
                    local_peaks_in_array = np.zeros_like(background_mask).astype(int)
                    for idx, point in enumerate(seeding_points, start=1):
                        local_peaks_in_array[point[0], point[1], point[2]] = idx
                    background_image = (self.image_data > background_threshold).astype(int)
                    distance = distance_transform_edt(background_image)
                    wts = watershed(-distance, local_peaks_in_array, mask=inverted_image)
                    print("wts.max()", wts.max())
                    wts *= background_image
                    # watershed_locs = find_watershed_locations(wts, 50)
                    # filtered_watershed_locs = filter_watershed_locations(watershed_locs, 50)
                    # #print("filtered_watershed_locs", filtered_watershed_locs)
                    # final_watershedded = refine_watershed(filtered_watershed_locs, wts.shape, 50)
                    # #final_watershedded = background_mask * background_image * final_watershedded
                    # locs = find_watershed_locations(final_watershedded, 50)
                    # current_index = 1
                    # added_new_cells = 0
                    # self.foreground_points = []
                    # for points in locs:
                    #     for point in points:
                    #         self.foreground_points.append((point[0], point[1], point[2], current_index, current_index%num_colors))
                    #     current_index += 1
                    #     added_new_cells += 1
                    #print("added_new_cells", added_new_cells)
                    self.cell_idx_display.update_text(self.index_control.cell_index, self.current_highest_cell_index)
                    self.index_control.update_index(self.index_control.cell_index, self.current_highest_cell_index)
                    self.update_index_display()
                    self.update_xy_view()
                    self.update_xz_view()
                    self.update_yz_view()
            elif mbox.clickedButton() == button2:
                print("Option 2 selected")
            elif mbox.clickedButton() == button3:
                print("Option 3 selected")
                self.watershed_neighborhood_selection_enabled = True
                self.central_widget.setCursor(self.droplet_cursor)
                QMessageBox.about(self, "Watershed Active", "Please select the center of a cell to be filled." 
                    + "\n You can control the neighborhood size with + and -, cancel with right click." 
                    + "\n Confirm your selection with enter.")
        else:
            QMessageBox.about(self, "No foreground cells", "Please label some cells first.")

    @pyqtSlot(int)
    def handle_progress(self, progress):
        self.progress_label.setVisible(True)
        self.progress_label.setText(f"Progress: {progress:.1f}%")

    @pyqtSlot(tuple)
    def handle_thresholds(self, result):
        # Handle the result from the worker thread
        centre_threshold, background_threshold = result
        self.cell_centre_threshold = round(centre_threshold)
        self.background_threshold = round(background_threshold)
        print("Centre Threshold: ", self.cell_centre_threshold )
        print("Background Threshold: ", self.background_threshold)

    @pyqtSlot()
    def handle_finished(self):
        self.progress_label.setVisible(False)

    def update_index_display(self):
        self.cell_idx_display.update_text(self.index_control.cell_index, self.current_highest_cell_index)
    
    def plot_hist(self):
        ax = self.hist_figure.add_subplot(111)
        ax.hist(self.image_data.ravel(), bins=int(self.max_pixel_intensity), range=(self.min_pixel_intensity, self.max_pixel_intensity))
        ax.set_title('Histogram of Pixel Intensities')
        ax.set_ylabel("Bin pixel count")
        ax.set_xlabel("Pixel intensity")
        self.hist_canvas.draw()

    def toggleForeground(self):
        if not self.foreground_enabled:
            self.drawing = True
            self.foreground_enabled = True
            self.background_enabled = False
            self.eraser_enabled = False
            self.foreground_button.setStyleSheet("background-color: lightgreen")
            self.background_button.setStyleSheet("")
            self.eraser_button.setStyleSheet("")
            self.central_widget.setCursor(self.brush_cursor)
        else:
            self.drawing = False
            self.foreground_enabled = False
            self.foreground_button.setStyleSheet("")
            self.background_button.setStyleSheet("")
            self.eraser_button.setStyleSheet("")
            self.central_widget.setCursor(self.default_cursor)
        self.repaint()
        self.central_widget.clearFocus()
        
    def toggleBackground(self):
        if not self.background_enabled:
            self.drawing = True
            self.foreground_enabled = False
            self.background_enabled = True
            self.eraser_enabled = False
            self.background_button.setStyleSheet("background-color: lightgreen")
            self.foreground_button.setStyleSheet("")
            self.eraser_button.setStyleSheet("")
            self.central_widget.setCursor(self.brush_cursor)
        else:
            self.drawing = False
            self.background_enabled = False
            self.background_button.setStyleSheet("")
            self.foreground_button.setStyleSheet("")
            self.eraser_button.setStyleSheet("")
            self.central_widget.setCursor(self.default_cursor)
        self.repaint()
        self.central_widget.clearFocus()
        
    def toggleEraser(self):
        self.foreground_enabled = False
        self.background_enabled = False
        if not self.eraser_enabled:
            self.eraser_enabled = True
            self.drawing = True
            self.eraser_button.setStyleSheet("background-color: lightgreen")
            self.foreground_button.setStyleSheet("")
            self.background_button.setStyleSheet("")
            self.central_widget.setCursor(self.default_cursor)
        else:
            self.eraser_enabled = False
            self.drawing = False
            self.eraser_button.setStyleSheet("")
            self.foreground_button.setStyleSheet("")
            self.background_button.setStyleSheet("")
        self.repaint()
        self.central_widget.clearFocus()
        
    def updateBrushWidthFromLineEdit(self):
        new_width_str = self.brush_text.text()
        try:
            new_width = int(new_width_str)
            if new_width <= 0:
                raise ValueError("Brush width must be a positive integer")
            self.updateBrushWidth(new_width)
        except ValueError:
            self.brush_text.setText(str(self.brush_width))
        finally:
            self.brush_text.clearFocus()

    def updateBrushWidth(self, new_width):
        self.brush_width = new_width

    def updateEraserRadius(self):
        new_radius_str = self.eraser_radius_text.text()
        try:
            new_radius = int(new_radius_str)
            if new_radius <= 0:
                raise ValueError("Eraser radius must be a positive integer")
            self.eraser_radius = new_radius
        except ValueError:
            self.eraser_radius_text.setText(str(self.eraser_radius))
        finally:
            self.eraser_radius_text.clearFocus()

    def remove_indices(self, indices):
        self.foreground_points = [p for i, p in enumerate(self.foreground_points) if i not in indices]
        self.update_xy_view()
        self.update_xz_view()
        self.update_yz_view()

    def removePoints(self, point, view_plane):
        if self.eraser_enabled and self.markers_enabled:
            if view_plane == "XY":
                self.foreground_points = [p for p in self.foreground_points \
                                                    if (math.dist((point[0], point[1]), 
                                                                (p[0], p[1])) > self.eraser_radius or p[2] != point[2]) or \
                                                                    p[-2] != self.index_control.cell_index]
                self.background_points = [p for p in self.background_points \
                                                    if (math.dist((point[0], point[1]), 
                                                                (p[0], p[1])) > self.eraser_radius or p[2] != point[2])]
            elif view_plane == "XZ":
                self.foreground_points = [p for p in self.foreground_points \
                                                    if (math.dist((point[0], point[2]), 
                                                                (p[0], p[2])) > self.eraser_radius or p[1] != point[1]) or \
                                                                    p[-2] != self.index_control.cell_index]
                self.background_points = [p for p in self.background_points \
                                                    if (math.dist((point[0], point[2]), 
                                                                (p[0], p[2])) > self.eraser_radius or p[1] != point[1])]
            elif view_plane == "YZ":
                self.foreground_points = [p for p in self.foreground_points \
                                                    if (math.dist((point[1], point[2]), 
                                                                (p[1], p[2])) > self.eraser_radius or p[0] != point[0]) or \
                                                                    p[-2] != self.index_control.cell_index]
                self.background_points = [p for p in self.background_points \
                                                    if (math.dist((point[1], point[2]), 
                                                                (p[1], p[2])) > self.eraser_radius or p[0] != point[0])]
    def check_overlaps(self, points):
        points_wo_index = [tuple(point[:-1]) for point in points]
        indices_to_remove = []
        for idx, point in enumerate(points):
            current_points_wo_index = points_wo_index[:idx] + points_wo_index[idx+1:]
            if tuple(point[:-1]) in current_points_wo_index:
                indices_to_remove.append(idx)
        if indices_to_remove:
            overlapnum = str(len(indices_to_remove))
            mbox = QMessageBox.question(self, 'Warning: {} Points overlap'.format(overlapnum), 
                                        "Some points overlap. Do you want to remove them?", 
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if mbox == QMessageBox.Yes:
                return [points[idx] for idx in range(len(points)) if idx not in indices_to_remove]
            else:
                return points
        else:
            return points
    
    def saveMasks(self):
        mask = np.zeros_like(self.image_data, dtype=int) 
        mbox = QMessageBox.question(self, 'Overlaps', "Do you want to check for overlaps? This may take some time", 
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if mbox == QMessageBox.Yes:
            points = self.check_overlaps(self.foreground_points)
            QMessageBox.about(self, "Overlaps checked", "Overlaps checked")
        points = self.foreground_points
        for point in self.background_points:
            x, y, z = point
            if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
                mask[z, y, x] = 0
        for point in points:
            x, y, z, idx, color_idx = point
            if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
                mask[z, y, x] = idx
        filenameNew = self.filename.replace(".", "")
        mask_name = filenameNew + "_mask.npy"
        if os.path.isfile(mask_name):
            mbox = QMessageBox.question(self, 'Warning: Masks file exists', "Do you want to overwrite it?", 
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if mbox == QMessageBox.Yes:
                np.save(mask_name, mask)
                QMessageBox.about(self, "Masks saved", "Masks saved as %s" % (mask_name))
            else:
                QMessageBox.about(self, "Saving aborted", "Masks not saved")
        else:
            np.save(mask_name, mask)
            QMessageBox.about(self, "Masks saved", "Masks saved as %s" % (mask_name))
        
    def load_masks(self, filename, load_background = False):
        # Show the loading screen
        self.loading_screen = LoadingScreen()
        self.loading_screen.show()

        # Create and start the worker thread
        self.mask_loader = MaskLoader(self, filename, load_background)
        self.mask_loader.finished.connect(self.on_masks_loaded)
        self.mask_loader.start()

    def on_masks_loaded(self):
        # Hide the loading screen when the task is done
        self.loading_screen.hide()
        self.cell_idx_display.update_text(self.index_control.cell_index, self.current_highest_cell_index)
        self.index_control.update_index(self.index_control.cell_index, self.current_highest_cell_index)
        self.update_index_display()
        self.update_xy_view()
        self.update_xz_view()
        self.update_yz_view()

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Mask", "", "Numpy Files (*.npy)", options=options)
        if file_name:
            mbox = QMessageBox.question(self, 'Background pixels', "Do you want to color the background (0s) also? (Not recommended)", 
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if mbox == QMessageBox.Yes:
                mbox = QMessageBox.question(self, 'Load Background pixels', "Are you sure?", 
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if mbox == QMessageBox.Yes:
                    self.load_masks(file_name, True)
                else:
                    self.load_masks(file_name, False)
            else:
                self.load_masks(file_name, False)

    def localContrastEnhancer(self):
        if self.local_contrast_enhancer_enabled:
            self.local_contrast_enhancer_enabled = False
            self.first_mouse_pos_for_contrast_rect = None
            self.last_mouse_pos_for_contrast_rect = None
            self.local_contrast_enhancer_button.setStyleSheet("")
            self.submit_contrast_button.hide()
        else:
            self.local_contrast_enhancer_enabled = True
            self.submit_contrast_button.show()
            self.local_contrast_enhancer_button.setText("Local Contrast Enhancer")
            self.local_contrast_enhancer_button.setStyleSheet("background-color: lightgreen")
        self.repaint()
        self.update_xy_view()

    def toggle_submit_contrast(self):
        if (self.first_mouse_pos_for_contrast_rect and self.last_mouse_pos_for_contrast_rect) and not self.visualization_only:
            x_min = min(self.first_mouse_pos_for_contrast_rect.x(), self.last_mouse_pos_for_contrast_rect.x())
            x_max = max(self.first_mouse_pos_for_contrast_rect.x(), self.last_mouse_pos_for_contrast_rect.x())
            y_min = min(self.first_mouse_pos_for_contrast_rect.y(), self.last_mouse_pos_for_contrast_rect.y())
            y_max = max(self.first_mouse_pos_for_contrast_rect.y(), self.last_mouse_pos_for_contrast_rect.y())
            x_diff = abs(self.last_mouse_pos_for_contrast_rect.x() - self.first_mouse_pos_for_contrast_rect.x())
            y_diff = abs(self.last_mouse_pos_for_contrast_rect.y() - self.first_mouse_pos_for_contrast_rect.y())
            z_height = max(x_diff, y_diff) // 2
            z_min = max(self.slider.value() - z_height, 0)
            z_max = min(self.slider.value() + z_height, self.z_max)
            subimage = self.image_data[z_min:z_max, y_min:y_max, x_min:x_max]
            eq_subimage = equalize_hist(subimage) * self.max_pixel_intensity
            self.image_data[z_min:z_max, y_min:y_max, x_min:x_max] = eq_subimage
            self.first_mouse_pos_for_contrast_rect = None
            self.last_mouse_pos_for_contrast_rect = None
            self.submit_contrast_button.hide()
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
            self.plot_hist()
        else:
            QMessageBox.about(self, "No selection chosen", "%s" % ("Please choose a region for contrast enhancement  \
                                                                   by using the local contrast button for drawing"))
    def findCell(self):
        pts = self.foreground_points
        if pts:
            if not self.visualization_only:
                last_cell_index = pts[-1][3]
                matching_points = [point[:4] for point in pts if point[3] == last_cell_index]
                if len(matching_points) > 2:
                    average_point = np.median(matching_points, axis=0)
                    self.slidery.setValue(round(average_point[1]))
                    self.sliderx.setValue(round(average_point[0]))
                    self.slider.setValue(round(average_point[2]))
                else:
                    self.slidery.setValue(pts[-1][1])
                    self.sliderx.setValue(pts[-1][0])
                    self.slider.setValue(pts[-1][2])
        else:
            QMessageBox.about(self, "Foreground empty", "%s" % ("Please draw cells using the foreground button"))

    def markersOffOn(self):
        
        if self.markers_enabled:
            self.markers_enabled = False
            self.markers_off_on_button.setText("Markers Off (M)")
            self.foreground_enabled = False
            self.background_enabled = False
            self.eraser_enabled = False 
            if self.visualization_only and self.backup_greyscale is not None:
                self.backup_color = self.image_data
                self.image_data = self.backup_greyscale
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
        else:
            self.markers_enabled = True
            self.markers_off_on_button.setText("Markers On (M)")
            if self.visualization_only and self.backup_color is not None:
                self.backup_greyscale = self.image_data
                self.image_data = self.backup_color
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
        self.repaint()

    def update_xy_view(self):
        z_index = self.slider.value()
        image = self.image_data[z_index]
        pixmap = self.numpyArrayToPixmap(image)
        if self.markers_enabled and not self.visualization_only:

            painter = QPainter(pixmap)
            if self.foreground_points:
                color_idx = self.foreground_points[0][-1]
                color_count = 0
                pen = QPen(QColor(glasbey_cmap[color_idx]))
                pen.setWidth(1)
                painter.setPen(pen)
                relevant_points = [point for point in self.foreground_points if point[2] == z_index]
                for point in relevant_points:
                    if point[-1] != color_idx:
                        color_idx = point[-1]
                        color_count += 1
                        pen = QPen(QColor(glasbey_cmap[color_idx]))
                        pen.setWidth(1)
                        painter.setPen(pen)
                    painter.drawPoint(point[0], point[1])
                    
            pen = QPen(Qt.blue)
            pen.setWidth(1)
            painter.setPen(pen)

            if self.background_points and not self.visualization_only:
                for point in self.background_points:
                    if point[2] == z_index:
                        painter.drawPoint(point[0], point[1])

            if self.view_finder:
                pen = QPen(QColor(255, 255, 0, 80))
                pen.setWidth(1)
                painter.setPen(pen)
                y_val = self.slider_to_pixmap(self.slidery.value(), 0, self.y_max, 0, pixmap.height())
                x_val = self.slider_to_pixmap(self.sliderx.value(), 0, self.x_max, 0, pixmap.width())
                square_size = 30 
                top_left_x = x_val - square_size // 2
                top_left_y = y_val - square_size // 2
                painter.drawEllipse(top_left_x, top_left_y, square_size, square_size)
                painter.drawLine(x_val, 0, x_val, pixmap.height())
                painter.drawLine(0, y_val, pixmap.width(), y_val)
            painter.end()

        if self.local_contrast_enhancer_enabled and \
            self.first_mouse_pos_for_contrast_rect and \
                self.last_mouse_pos_for_contrast_rect and \
                    not self.visualization_only:
            painter = QPainter(pixmap)
            pen = QPen(Qt.red)
            pen.setWidth(1)
            painter.setPen(pen)
            rect = QRect(self.first_mouse_pos_for_contrast_rect, self.last_mouse_pos_for_contrast_rect)
            painter.drawRect(rect)
            painter.end()

        if self.watershed_neighborhood_selection_enabled and self.watershed_seeding_point:
            xval = self.watershed_seeding_point[0]
            yval = self.watershed_seeding_point[1]
            radius = self.watershed_radius
            if xval - radius >= 0 and yval - radius >= 0 and xval + radius < self.x_max and yval + radius < self.y_max:
                painter = QPainter(pixmap)
                pen = QPen(Qt.red)
                pen.setWidth(1)
                painter.setPen(pen)
                front_top_left = QPointF(xval - radius, yval - radius)
                front_bottom_right = QPointF(xval + radius, yval + radius)
                painter.drawRect(QRectF(front_top_left, front_bottom_right))
                painter.end()

        if len(self.watershed_foreground_points) > 0:
            painter = QPainter(pixmap)
            pen = QPen(Qt.red)
            pen.setWidth(1)
            painter.setPen(pen)
            for point in self.watershed_foreground_points:
                if point[2] == z_index:
                    painter.drawPoint(point[0], point[1])
                #painter.drawPoint(point[1], point[0])

            painter.end()
        self.xy_view.setPixmap(pixmap)

    def slider_to_pixmap(self, slider_value, slider_min, slider_max, pixmap_min, pixmap_max):
        return int((slider_value - slider_min) / (slider_max - slider_min) * (pixmap_max - pixmap_min) + pixmap_min)

    def update_xz_view(self):   
        y_index = self.slidery.value()
        if self.image_data.ndim == 4:
            image = self.image_data[:, y_index, :, :]
            image = np.transpose(image, (1, 0, 2))
        else:
            image = self.image_data[:, y_index, :].T
        pixmap = self.numpyArrayToPixmap(image)

        if self.markers_enabled and not self.visualization_only:
            painter = QPainter(pixmap)
            if self.foreground_points:
                color_idx = self.foreground_points[0][-1]
                color_count = 0
                pen = QPen(QColor(glasbey_cmap[color_idx]))
                pen.setWidth(1)
                painter.setPen(pen)
                relevant_points = [point for point in self.foreground_points if point[1] == y_index]

                for point in relevant_points:
                    if point[-1] != color_idx:
                        color_idx = point[-1]
                        color_count += 1
                        pen = QPen(QColor(glasbey_cmap[color_idx]))
                        pen.setWidth(1)
                        painter.setPen(pen)
                    painter.drawPoint(point[2], point[0])


            pen = QPen(Qt.blue)
            pen.setWidth(1)
            painter.setPen(pen)

            if self.background_points and not self.visualization_only:
                for point in self.background_points:
                    if point[1] == y_index:
                        painter.drawPoint(point[2], point[0])
            if self.view_finder:
                pixmapx = self.slider_to_pixmap(self.slider.value(), 0, self.z_max, 0, pixmap.width())
                pixmapy = self.slider_to_pixmap(self.sliderx.value(), 0, self.x_max, 0, pixmap.height())
                pen = QPen(QColor(255, 255, 0, 80))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawLine(pixmapx, 0, pixmapx, pixmap.height())
                painter.drawLine(0, pixmapy, pixmap.width(), pixmapy)

            painter.end()

        if len(self.watershed_foreground_points) > 0:
            painter = QPainter(pixmap)
            pen = QPen(Qt.red)
            pen.setWidth(1)
            painter.setPen(pen)
            for point in self.watershed_foreground_points:
                if point[1] == y_index:
                    painter.drawPoint(point[2], point[0])

            painter.end()
        self.xz_view.setPixmap(pixmap)

    def update_yz_view(self):
        x_index = self.sliderx.value()
        if self.image_data.ndim == 4:
            image = self.image_data[:, :, x_index, :]
            image = np.transpose(image, (1, 0, 2))
        else:
            image = self.image_data[:, :, x_index].T
        pixmap = self.numpyArrayToPixmap(image)
        if self.markers_enabled and not self.visualization_only:
            painter = QPainter(pixmap)
            if self.foreground_points:
                color_idx = self.foreground_points[0][-1]
                color_count = 0
                pen = QPen(QColor(glasbey_cmap[color_idx]))
                pen.setWidth(1)
                painter.setPen(pen)
                relevant_points = [point for point in self.foreground_points if point[0] == x_index]
                
                for point in relevant_points:
                    if point[-1] != color_idx:
                        color_idx = point[-1]
                        color_count += 1
                        pen = QPen(QColor(glasbey_cmap[color_idx]))
                        pen.setWidth(1)
                        painter.setPen(pen)
                    painter.drawPoint(point[2], point[1])


            pen = QPen(Qt.blue)
            pen.setWidth(1)
            painter.setPen(pen)

            if self.background_points and not self.visualization_only:
                for point in self.background_points:
                    if point[0] == x_index:
                        painter.drawPoint(point[2], point[1])

            if self.view_finder:    
                pixmapy = self.slider_to_pixmap(self.slidery.value(), 0, self.y_max, 0, pixmap.height())
                pixmapx = self.slider_to_pixmap(self.slider.value(), 0, self.z_max, 0, pixmap.width())
                pen = QPen(QColor(255, 255, 0, 80))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawLine(pixmapx, 0, pixmapx, pixmap.height())
                painter.drawLine(0, pixmapy, pixmap.width(), pixmapy)

            painter.end()

        if len(self.watershed_foreground_points) > 0:
            painter = QPainter(pixmap)
            pen = QPen(Qt.red)
            pen.setWidth(1)
            painter.setPen(pen)
            for point in self.watershed_foreground_points:
                if point[0] == x_index:
                    painter.drawPoint(point[2], point[1])

            painter.end()
        self.yz_view.setPixmap(pixmap)

def main():
    app = QApplication(sys.argv)
    parser = argparse.ArgumentParser(description='Load an image.')
    parser.add_argument('filename', type=str, help='Path to the image to load')
    args = parser.parse_args()
    filename = args.filename
    if filename is None:
        parser.error("Please provide a filename.")
        sys.exit(1)
    window = MainWindow(filename)
    window.setGeometry(100, 100, 800, 600)
    window.setWindowTitle(f'3D TIFF Viewer - {filename}')
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

