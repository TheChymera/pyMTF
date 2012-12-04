#!/usr/bin/env python
from __future__ import division
__author__ = 'Horea Christian'

import Image
import gtk
import numpy as np
from pylab import figure, show, errorbar
import matplotlib.pyplot as plt
from matplotlib import axis

if gtk.pygtk_version < (2,3,90):
    print "PyGtk 2.3.90 or later required for Plot-It"
    raise SystemExit
dialog = gtk.FileChooserDialog("Choose a set of lens chart pictures...",
                               None,
                               gtk.FILE_CHOOSER_ACTION_OPEN,
                               (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                gtk.STOCK_OPEN, gtk.RESPONSE_OK))
dialog.set_default_response(gtk.RESPONSE_OK)
dialog.set_select_multiple(True)

lefilter = gtk.FileFilter()
lefilter.set_name("Image files")
lefilter.add_pattern("*.jpg")
lefilter.add_pattern("*.JPG")
lefilter.add_pattern("*.jpeg")
lefilter.add_pattern("*.JPEG")
lefilter.add_pattern("*.png")
lefilter.add_pattern("*.PNG")
dialog.add_filter(lefilter)

response = dialog.run()
if response == gtk.RESPONSE_OK:
    data_names = dialog.get_filenames()
    print dialog.get_filenames(), 'selected'
elif response == gtk.RESPONSE_CANCEL:
    print 'Closed, no files selected'
dialog.destroy()

ranking = np.r_['1,3,0', data_names, np.zeros(np.shape(data_names)), np.zeros(np.shape(data_names))]
ranking = np.reshape(ranking, (len(data_names),3))

img_exif=Image.open(data_names[0])
exifs = img_exif._getexif()
lens_focal_length = np.array(exifs.get(0x920A))[0]/np.array(exifs.get(0x920A))[1] # dodgy way to identify the lens. Apparently the PIL exif function can't read the lens-relevant tags
lens_name = str(lens_focal_length) + 'mm '

for i, v in enumerate(data_names):
    img = Image.open(v)
    exif = img._getexif()
    img = np.asarray(img)
    orient = exif.get(0x0112) # get exif orientation key (0x0112 = 274)
    aperture = np.array(exif.get(0x829D)) # get eixf aperture key (nikon d5100 for which this was initially written uses 0x829D instead of the usual 0x9202)
    aperture = aperture[0] / aperture[1] # the exif aperture key contains a whole number (aperture*10) and a 10 (nikon d5100 only?)
    camera_model = exif.get(0x110)
        
    if np.ndim(img) == 3:           #make grayscale, do nothing if image array is not 3D
        img_gray = img.sum(axis=2)
    else: img_gray = img
    
    if orient == 6:                 #adjust image orientation according to eixf tags
        img_gray = np.rot90(img_gray, 3)
    elif orient == 3:
        img_gray = np.rot90(img_gray, 2)   
    elif orient == 8:   
        img_gray = np.rot90(img_gray, 1) 
    elif orient == 2:
        img_gray = np.fliplr(img_gray)
    elif orient == 4:
        img_gray = np.flipud(img_gray)
    elif orient == 5:
        img_gray = np.fliplr(np.rot90(img_gray, 1))
    elif orient == 7:
        img_gray = np.fliplr(np.rot90(img_gray, 3))
    else: img_gray = img_gray    

    height = np.shape(img_gray)[0] 
    width = np.shape(img_gray)[1]
    hline = img_gray[height/2, width/8 : 7*width/8] #determine horizontal midline - where the lens chart should be most probably located 
    hline = (hline-hline.min())/hline.max() # normalize hline (supposedly to correct for varying brightness :-?)

    #determine dips (negative peaks) - dirty workaround based on single surrounding values:    
    dips = np.r_[False, hline[1:] < hline[:-1]] & np.r_[hline[:-1] < hline[1:], False] & np.r_[False, False, hline[2:] < hline[:-2]] & \
        np.r_[hline[:-2] < hline[2:], False, False] & np.r_[False, False, False, hline[3:] < hline[:-3]] & np.r_[hline[:-3] < hline[3:], False, False, False] &\
        np.r_[False, False, False, False, hline[4:] < hline[:-4]] & np.r_[hline[:-4] < hline[4:], False, False, False, False] &\
        np.r_[False, False, False, False, False, hline[5:] < hline[:-5]] & np.r_[hline[:-5] < hline[5:], False, False, False, False, False] &\
        np.r_[False, False, False, False, False, False, hline[6:] < hline[:-6]] & np.r_[hline[:-6] < hline[6:], False, False, False, False, False, False] 
    un_sharpness = np.sum(dips*hline) # since these are dips the greater the value the smaller the sharpness
    ranking[i,1:] = un_sharpness
    ranking[i,2:] = aperture

ranking = np.array(ranking[:,1:], dtype=np.float) # first column ignored (no numeric values), converted to float
rnk_sort = ranking[ranking[:,1].argsort()] #sorted by aperture- argsort gives a row number's list so that the column is ascending  
last=rnk_sort[:,1] # only the aperture values
w = np.where(last[:-1] != last[1:])[0] + 1 #gives the ordinal element number for which their value in last[:-1] (all but last el) is not the same as in
#last[1:] ; [0] converts it to a np.array ; and +1 accounts for the fact that the first number from np.where is 4 (because numbering starts with 0)
w = np.concatenate(([0], w, [len(rnk_sort)])) #add 0 and last value
means = np.add.reduceat(rnk_sort, w[:-1])/np.diff(w)[:,None]
#bins together the values of rnk_sort dividing them at the positions specified in w, then divides that by the difference between the incremental w values
stds = np.add.reduceat(rnk_sort**2, w[:-1])/np.diff(w)[:,None] - means**2 #modified computations from above to obtain std
stds = stds[:,0]**0.5
ind = np.arange(len(means[:,0]))

fig = figure(figsize=(ind.max(), 6), dpi=80,facecolor='#eeeeee')
ax=fig.add_subplot(1,1,1)
width = 0.3
ax.yaxis.grid(True, linestyle='-', which='major', color='#dddddd',alpha=0.5, zorder = 1)
box_plots = plt.bar(ind, means[:,0], width ,color='#cccccc', alpha=0.35, zorder = 2)
bars = errorbar(ind+(width/2 - 0.0), means[:,0], yerr=stds, color='k', ecolor='r', elinewidth='4', capsize=0, linestyle='None', zorder = 3)
ax.set_xlim(0, ind.max())
ax.set_ylabel('Semi-quantitative Un-sharpness Score')
ax.set_xlabel('Aperture')
ax.set_xticks(ind + width/2)
ax.set_title('Unsharpness of '+ camera_model +' with '+ lens_name+' lens')
ax.set_xticklabels(means[:,1])
for tick in ax.axes.get_xticklines():
    tick.set_visible(False)
axis.Axis.zoom(ax.xaxis, -0.3)
show()