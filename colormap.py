from PIL import ImageColor

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
glasbey_cmap = glasbey_cmap_100
glasbey_cmap_rgb = [ImageColor.getcolor(col, "RGB") for col in glasbey_cmap]
num_colors = len(glasbey_cmap)

# glasbey_cmap = glasbey_cmap[:10]

# tab10_colormap = matplotlib.colormaps['tab10']
# glasbey_cmap = [mcolors.to_hex(color) for color in tab10_colormap.colors]
# del glasbey_cmap[7] # get rid of the grey one
