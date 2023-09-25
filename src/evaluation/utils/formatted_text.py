import sys
import fire
import matplotlib



def esc(code):
    return f'\033[{code}m'

def esc_rgb(r, g, b):
    return esc(f'48;2;{r};{g};{b}')

def to_color(value, cmap_name='cool'):
    # Select a sequential colormap from matplotlib
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html#sequential
    cmap = matplotlib.cm.get_cmap(cmap_name)

    rgba = cmap(value, bytes=True)
    return rgba

def color_text(text, rgb=(255, 255, 0)):
    sys.stdout.write(esc_rgb(*rgb) + text + esc('0'))

def test():
    color_text('Hi')
    color_text(' there!', rgb=(255, 0, 255))
    print()

def helloworld():
    # Print text with a red background
    sys.stdout.write(esc('41') + 'Hello, world!' + esc('0'))
    print()


if __name__ == '__main__':
  fire.Fire()