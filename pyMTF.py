#!/usr/bin/env python
from __future__ import division
__author__ = 'Horea Christian'

import Image
import gtk, math
import numpy as np
from pylab import figure, show
import matplotlib.pyplot as plt

if gtk.pygtk_version < (2,3,90):
    print "PyGtk 2.3.90 or later required for Plot-It"
    raise SystemExit
dialog = gtk.FileChooserDialog("Choose a block file...",
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
    print dialog.get_filename(), 'selected'
elif response == gtk.RESPONSE_CANCEL:
    print 'Closed, no files selected'
dialog.destroy()

ranking = np.r_['1,3,0', data_names, np.zeros(np.shape(data_names)), np.zeros(np.shape(data_names))]
ranking = np.reshape(ranking, (len(data_names),3))

for i, v in enumerate(data_names):
    img = Image.open(v)
    exif = img._getexif()
    img = np.asarray(img)
    orient = exif.get(0x0112) # get exif orientation key (0x0112 = 274) 0x9202
    aperture = np.array(exif.get(0x829D)) # get eixf aperture key (nikon d5100 for which this was initially written uses 0x829D instead of the canonical 0x9202)
    aperture = aperture[0] / aperture[1] # the exif aperture key contains a whole number (aperture*10) and a 10 (nikon d5100 only?)
        
    if np.ndim(img) == 3:           #make grayscale, do nothing if image array is not 3D
        img_gray = img.sum(axis=2)
    else: img_gray = img
    
    if orient == 6:                 #adjust image orientation according to eixf tags
        img_gray = np.rot90(img_gray, 3)
    elif orient == 3:
        img_gray = np.rot90(img_gray, 2)   
    elif orient == 8:
        img_gray = np.rot90(img_gray, 1)   
    else: img_gray = img_gray    

    height = np.shape(img_gray)[0] 
    width = np.shape(img_gray)[1]
    hline = img_gray[height/2, width/8 : 7*width/8] #determine horizontal midline - where the lens chart should be most probably located 
    hline = hline/hline.max() # normalize hline (supposedly to correct for varying brightness :-?)

    #determine peaks - dirty workaround based on single surrounding values:    
    peaks = np.r_[False, hline[1:] < hline[:-1]] & np.r_[hline[:-1] < hline[1:], False] & np.r_[False, False, hline[2:] < hline[:-2]] & \
        np.r_[hline[:-2] < hline[2:], False, False] & np.r_[False, False, False, hline[3:] < hline[:-3]] & np.r_[hline[:-3] < hline[3:], False, False, False] &\
        np.r_[False, False, False, False, hline[4:] < hline[:-4]] & np.r_[hline[:-4] < hline[4:], False, False, False, False] &\
        np.r_[False, False, False, False, False, hline[5:] < hline[:-5]] & np.r_[hline[:-5] < hline[5:], False, False, False, False, False] &\
        np.r_[False, False, False, False, False, False, hline[6:] < hline[:-6]] & np.r_[hline[:-6] < hline[6:], False, False, False, False, False, False] 
    sharpness = np.sum(peaks*hline)
    ranking[i,1:] = sharpness
    ranking[i,2:] = aperture

print ranking    
ranking = np.array(ranking[:,1:], dtype=np.float)
sorted = ranking[ranking[:,1].argsort()]
print sorted
last=sorted[:,1]
w = np.where(last[:-1] != last[1:])[0] + 1
w = np.concatenate(([0], w, [len(sorted)]))
means = np.add.reduceat(sorted, w[:-1])/np.diff(w)[:,None]
stds = np.add.reduceat(sorted**2, w[:-1])/np.diff(w)[:,None] - means**2
stds = stds[:,0]**0.5
print means, stds

fig = figure(facecolor='#eeeeee')
ax=fig.add_subplot(1,1,1)
ind = np.arange(len(means[:,0]))
width = 0.3
bars = plt.bar(ind, means[:,0], width ,color='k', alpha=0.25, yerr=stds, ecolor='r')
ax.set_ylabel('Semi-quantitative Sharpness Score')
ax.set_title('Aperture')
ax.set_xticks(ind+width)
ax.set_xticklabels(means[:,1])
show()