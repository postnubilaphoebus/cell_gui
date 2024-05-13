## Python GUI for annotating cells <br>  <br>
This is still a work in progress and mostly for personal use thus far.

### Installation <br>
pip install -r requirements.txt <br>  <br>

### Usage  <br>
Open an image (.tif or .npy) with the GUI using `python gui.py "path/to/your/image" <br>
For example: python gui.py "img2.tif" <br>
Once in the GUI, you can open previously saved Masks that are saved as npy files. <br>
When a button has a keybinding associated with it, it is indicated on the button. <br>
Note that you are only working on one cell until you decide to move on to the next one or go to the previous one with the cell counter (<>). That is, you can only add to or erase elements of the current cell. <br>
The first image is the xy-view, the second image the xz-view, the third the yz-view. Annotating on the xy-view is recommended, you can then double-check on the other views if there are some inconsistencies. <br>
Some useful functions: <br>
*Find Cell*: Goes to the center of the cell with the highest cell index that was last annotated. Useful if your brain has melted from too much annotation and you want to check if the new color of the brush is different. <br>
*Left click*: used for annotating cells; Right click: copy exact coordinates from last slide (must be the same view). Right clicking again pastes the coordinates from the last slide. Copy-paste with right-click works regardless of what cell you are working on. <br>
*Select Cell*: Sometimes you may want to correct an old cell. You can press "S" and then left-click on a cell, which means it is now editable. The "Selected Cell Index" and "Highest Cell Index" reflect that.  <br>
*Local Contrast Enhancement*: In case of a particularly dark/unclear region, you may apply local contrast enhancement. Draw a rectangle on the image after pressing the LCE button, and then press submit. A cuboid in the image volume is now locally enhanced using CLAHE.  <br>
All the other functions should be self-explanatory.
