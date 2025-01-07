import matplotlib.pyplot as plt
import matplotlib
font = {
        'family' : 'sans-serif',
        'size'   : 6,  #  # 修改为6，满足5-7 pt的要求
}
mathtext = {
        'fontset' : 'stixsans',#'stixsans',#'stix',
}
lines = {
        'linewidth' : 1.5,
}
xtick = {
        'direction' : 'out',
        'major.size' : 3,
        'major.width' : 1,
        'minor.size' : 2,
        'minor.width' : 1,
        'labelsize' : 6,  # 修改为6，满足5-7 pt的要求
}
ytick = {
        'direction' : 'out',
        'major.size' : 3,
        'major.width' : 1,
        'minor.size' : 2,
        'minor.width' : 1,
        'labelsize' : 6,  # 修改为6，满足5-7 pt的要求
}
axes = {
        'linewidth' : 1,
        'titlesize' : 6, # 修改为6，满足5-7 pt的要求
        #'titleweight': 'bold',
        'labelsize' :  6, # 修改为6，满足5-7 pt的要求
        #'labelweight' : 'bold',
}

matplotlib.rc('font',**font)
matplotlib.rc('mathtext',**mathtext)
matplotlib.rc('lines',**lines)
matplotlib.rc('xtick',**xtick)
matplotlib.rc('ytick',**ytick)
matplotlib.rc('axes',**axes)
width = 100 / 25.4 
# 记住最后一步要tight_layout!
# 创建一个新的figure，设置其DPI和尺寸
# fig, ax = plt.subplots(figsize=(180/25.4, 180/25.4), dpi=300)  # 180mm为最大宽度,这里的180/25.4是将mm转换为英寸，因为matplotlib的单位是英寸
# ax.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 16])  # 绘制一个简单的图形
# plt.savefig('figure.png', dpi=300, bbox_inches='tight')