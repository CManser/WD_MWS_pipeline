import MWS_WD_scripts
import numpy as np
import sys
import matplotlib.pyplot as plt 
import os
import scipy.interpolate

c = 299792.458 # Speed of light in km/s
stdwave = np.arange(3000,11000.1, 0.1)

inp = sys.argv[1]


name = inp.split('/')[-1]
file_path = inp[:-1*len(name)]

print('\n' + name)

filename = inp
name_r, name_z = filename.replace('-b', '-r'), filename.replace('-b', '-z')
spectra_b = np.loadtxt(filename,usecols=(0,1,2),unpack=True).transpose()
spectra_b[:,2] = spectra_b[:,2]**-0.5
spectra_r = np.loadtxt(name_r,usecols=(0,1,2),unpack=True).transpose()
spectra_r[:,2] = spectra_r[:,2]**-0.5
spectra_z = np.loadtxt(name_z,usecols=(0,1,2),unpack=True).transpose()
spectra_z[:,2] = spectra_z[:,2]**-0.5

training_set = MWS_WD_scripts.load_training_set()
WD_type =  MWS_WD_scripts.WD_classify(spectra_b, spectra_r, training_set = training_set)
print(WD_type[0])

best_T, best_T_err, best_g, best_g_err, best_rv, s_best_T, s_best_T_err, s_best_g, s_best_g_err, StN = MWS_WD_scripts.fit_DA(spectra_b[(np.isnan(spectra_b[:,1])==False) & (spectra_b[:,0]>3500)], plot = True, verbose_output = True)

modwave = stdwave*(best_rv+c)/c
model = MWS_WD_scripts.interpolating_model_DA(best_T,best_g/100, fine_models = True)
model2 = MWS_WD_scripts.interpolating_model_DA(s_best_T,s_best_g/100, fine_models = True)

# Plotting
fig = plt.figure(figsize= (11,9))
axes1 = fig.add_axes([0,0.45,1,0.55])
axes2 = fig.add_axes([0,0,1,0.4])

axes1.text(0.45, 0.95, name[:-6], transform=axes1.transAxes, fontsize = 14)
axes1.text(0.45, 0.90, 'T = {:.1f} +/- {:.1f} K  |  logg = {:.3f} +/- {:.3f}'.format(best_T, best_T_err, best_g/100, best_g_err/100), transform=axes1.transAxes, fontsize = 14)
axes1.text(0.45, 0.85, 'T2 = {:.1f} +/- {:.1f} K  |  logg2 = {:.3f} +/- {:.3f}'.format(s_best_T, s_best_T_err, s_best_g/100, s_best_g_err/100), transform=axes1.transAxes, fontsize = 14)
axes1.text(0.45, 0.80, 'rv = {:.2f} km/s  |  S/N = {:.1f}'.format(best_rv, StN), transform=axes1.transAxes, fontsize = 14)

axes1.plot(spectra_b[:,0], spectra_b[:,1], color = '0.2', lw = 1.0)
axes1.plot(spectra_r[:,0], spectra_r[:,1], color = '0.3', lw = 1.0)
axes1.plot(spectra_z[:,0], spectra_z[:,1], color = '0.3', lw = 1.0)

check_f_spec=spectra_b[:,1][(spectra_b[:,0]>4500.) & (spectra_b[:,0]<4700.)]
model[np.isnan(model)] = 0.0
check_f_model = model[(modwave > 4500) & (modwave < 4700)]
adjust = np.average(check_f_model)/np.average(check_f_spec)
axes1.plot(modwave[(modwave > 3600.0) & (modwave < 10500.0)], model[(modwave > 3600.0) & (modwave < 10500.0)]/adjust, color = 'red', alpha = 0.9, lw = 0.8)

model2[np.isnan(model2)] = 0.0
check_f_model2 = model2[(modwave > 4500) & (modwave < 4700)]
adjust2 = np.average(check_f_model2)/np.average(check_f_spec)
axes1.plot(modwave[(modwave > 3600.0) & (modwave < 10500.0)], model2[(modwave > 3600.0) & (modwave < 10500.0)]/adjust2, color = 'blue', alpha = 0.7, lw = 0.8)




axes1.set_ylabel('Flux',fontsize=12)
axes2.set_xlabel('Wavelength (Angstroms)',fontsize=12)

axes1.set_xlim(3500, 10600)
axes2.set_xlim(3500, 10600)
axes2.set_ylim(0.2, 1.8)

func = scipy.interpolate.interp1d(modwave, model)
model_b = func(spectra_b[:,0])
model_r = func(spectra_r[:,0])
model_z = func(spectra_z[:,0])

func2 = scipy.interpolate.interp1d(modwave, model2)
model2_b = func2(spectra_b[:,0])
model2_r = func2(spectra_r[:,0])
model2_z = func2(spectra_z[:,0])



axes2.plot(spectra_b[:,0], spectra_b[:,1]/model_b*adjust, color = 'red', alpha = 0.9, lw = 0.5)
axes2.plot(spectra_r[:,0], spectra_r[:,1]/model_r*adjust, color = 'red', alpha = 0.9, lw = 0.5)
axes2.plot(spectra_z[:,0], spectra_z[:,1]/model_z*adjust, color = 'red', alpha = 0.9, lw = 0.5)
axes2.plot(spectra_b[:,0], spectra_b[:,1]/model2_b*adjust2, color = 'blue', alpha = 0.4, lw = 0.5)
axes2.plot(spectra_r[:,0], spectra_r[:,1]/model2_r*adjust2, color = 'blue', alpha = 0.4, lw = 0.5)
axes2.plot(spectra_z[:,0], spectra_z[:,1]/model2_z*adjust2, color = 'blue', alpha = 0.4, lw = 0.5)

axes2.axhline(1, ls = '--', lw = 0.5, color = '0.3')

save_path = file_path + '/fits/'
if not os.path.isdir(save_path):
  try: 
    os.mkdir(save_path)
  except OSError:
    print('Could not make path {:s}'.format(save_path))
  else:
    print ("Successfully created the directory {:s}".format(save_path))
plt.savefig('{:s}{:s}_fitted.pdf'.format(save_path,name[:-6]), bbox_inches = 'tight')
plt.close()

if (StN > 10.0) & (WD_type == 'DA'):
  compare_b = np.vstack((spectra_b[:,0], spectra_b[:,1]/model_b*adjust, spectra_b[:,2]/model_b*adjust)).transpose()
  compare_r = np.vstack((spectra_r[:,0], spectra_r[:,1]/model_r*adjust, spectra_r[:,2]/model_r*adjust)).transpose()
  compare_z = np.vstack((spectra_z[:,0], spectra_z[:,1]/model_z*adjust, spectra_z[:,2]/model_z*adjust)).transpose()
  np.savetxt('{:s}{:s}-b_compare.dat'.format(save_path,name[:-6]), compare_b)
  np.savetxt('{:s}{:s}-r_compare.dat'.format(save_path,name[:-6]), compare_r)
  np.savetxt('{:s}{:s}-z_compare.dat'.format(save_path,name[:-6]), compare_z)

#plt.show()
