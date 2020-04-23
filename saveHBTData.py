# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:57:40 2020

@author: yumou

Retrieve and save HBT shot data to /Data folder.
Script will generate 2 files:
    1. FILENAME_data.csv
    2. FILENAME_shotnoloc.csv
    3. FILENAME_summary.txt
"""
import numpy as np
import pandas as pd
import os
import time

import hbtepLib as hbt
import getTimePoint

def main():
    #Enter run settings
    
    #Shot numbers
    shotno_start = int(input('Enter shotno_start: '))
    shotno_end = int(input('Enter shotno_end: '))
    if shotno_start <= 0 or shotno_end <= 0:
        print('ERROR: shotno_start & shotno_end must be positive')
        return False
    if shotno_end < shotno_start:
        print('ERROR: shotno_end must be larger than or equal to shotno_start')
        return False
    limit = int(input('Number of shots in total: '))
    #Downsampling
    downsampling = input('Enter downsampling rate (ENTER=no downsampling): ')
    if downsampling == '':
        downsampling = 1
    else:
        downsampling = int(downsampling)
        if downsampling < 1:
            print('ERROR: downsampling rate cannot be smaller than 1')
            return False 
    
    #Shot style filtering          -- TO_DO
    filtering = False
    
    #Select signals                -- TO_DO
    signals = ['ip','q','mr','lv','spect','sxrmid','n1amp','bpli','oh','vf']
    
    #Filename
    filename = str(input('Enter file name: '))
    
    
    
    print('----------------------------------')
    print('Run settings summary:')
    print('Shotno range: '+str(shotno_start)+'-'+str(shotno_end))
    print('Downsampling rate: ' + str(downsampling))
    print('Shot style filtering rule: ' + str(filtering))
    print('Signals: '+str(signals))
    print('File name: '+ filename)
    print('Date: ' + time.ctime())
    print('----------------------------------')
    
    ##########################################################################
    time_start = time.time()
    PATH_DIR = os.path.join(os.getcwd(), 'Data')
    PATH_DATA = os.path.join(PATH_DIR, filename+'_data.csv')
    PATH_SHOTNOLOC = os.path.join(PATH_DIR, filename+'_shotnoloc.csv')
    PATH_SUMMARY = os.path.join(PATH_DIR, filename+'_summary.txt')
    
    counter_shotno = 0
    counter_saved = 0
    shotno_total = shotno_end - shotno_start + 1
    
    data_shotno_row_index = np.empty((0,3))
    data_shot_row_index_start = 0
    data_shot_row_index_stop = 0
    
    
    column_name = ['shotno','t', 't-disrupt'] + signals
    
    for shotno in np.arange(shotno_start, shotno_end+1):
        shotno = int(shotno)
        counter_shotno += 1
        
        #Find breakdown & current spike time
        i_start = getTimePoint.getBreakdownTime(shotno, plot=False, return_index=True)
        i_stop = getTimePoint.getDisruptionTime(shotno, plot=False, return_index=True)
        if i_start == False or i_stop == False or i_stop - i_start < 1500:
            continue
        
        #Check filtering rule -- TO_DO
        
        ##########################################################
        #GET DATA
        data_np = np.empty((i_stop-i_start, len(column_name)))
        #shotno
        data_np[:,0] = shotno
        
        try: 
            #time array
            hbtt = hbt.get.ipData(shotno, tStop=0.02, findDisruption=False)
            t_start = hbtt.time[i_start]
            t_stop = hbtt.time[i_stop]
            data_np[:,1] = hbtt.time[i_start:i_stop]
            data_np[:,2] = t_stop - data_np[:,1]
            
            if 'ip' in signals:
                hbtip = hbt.get.ipData(shotno, tStart=t_start, tStop=t_stop, findDisruption=False)
                data_np[:,column_name.index('ip')] = hbtip.ip
                
            if 'q' in signals:
                hbtq = hbt.get.qStarData(shotno, tStart=t_start, tStop=t_stop)
                data_np[:,column_name.index('q')] = hbtq.qStar
            
            if 'mr' in signals:
                hbtmr = hbt.get.plasmaRadiusData(shotno, tStart=t_start, tStop=t_stop)
                data_np[:,column_name.index('mr')] = hbtmr.majorRadius
                
            if 'lv' in signals:
                hbtlv = hbt.get.loopVoltageData(shotno, tStart=t_start, tStop=t_stop)
                data_np[:,column_name.index('lv')] = hbtlv.loopVoltage
            
            if 'spect' in signals:
                hbtspect = hbt.get.spectrometerData(shotno, tStart=t_start, tStop=t_stop)
                data_np[:,column_name.index('spect')] = hbtspect.spect
            
            if 'sxrmid'in signals:
                hbtsxrmid = hbt.get.sxrMidplaneData(shotno, tStart=t_start, tStop=t_stop)
                
                #hbtsxrmid = hbt.get.sxrMidplaneData(shotno, tStart=t_start, tStop=t_stop)
                data_np[:,column_name.index('sxrmid')] = hbtsxrmid = hbtsxrmid.sxr
            
            if 'n1amp' in signals:
                hbtn1 = hbt.get.nModeData(shotno, tStart=t_start, tStop=t_stop)
                data_np[:,column_name.index('n1amp')] = hbtn1.n1Amp
                
            if 'bpli' in signals:
                hbtbpli = hbt.get.polBetaLi(shotno, tStart=t_start, tStop=t_stop)
                data_np[:,column_name.index('bpli')] = hbtbpli.polBetaLi
            
            if 'oh' in signals:
                hbtoh = hbt.get.capBankData(shotno, tStart=t_start, tStop=t_stop)
                data_np[:,column_name.index('oh')] = hbtoh.ohBankCurrent
            
            if 'vf' in signals:
                hbtvf = hbt.get.capBankData(shotno, tStart=t_start, tStop=t_stop)
                data_np[:,column_name.index('vf')] = hbtvf.vfBankCurrent
        except Exception:
            continue

        ##########################################################
        #Downsample data_np
        if downsampling > 1:
            data_np = data_np[0::downsampling,:]
            
        #Convert to dataframe & write to csv file
        data_df = pd.DataFrame(data_np, columns=column_name)
        if counter_saved != 0:  #Avoid saving multiple headers
            data_df.to_csv(PATH_DATA, mode='a', header=False, index=False)
        else:
            data_df.to_csv(PATH_DATA, mode='w', index=False)
        
        
        # add shotno, starting index
        data_shot_row_index_stop = data_shot_row_index_start + data_np.shape[0]
        data_shotno_row_index = np.append(data_shotno_row_index,
                                          [[shotno, data_shot_row_index_start,
                                            data_shot_row_index_stop]], axis=0)
        data_shot_row_index_start = data_shot_row_index_stop

        print(str(shotno) + ' saved ('+str(counter_shotno)+'/'+str(shotno_total)+')')        
        counter_saved += 1
        if counter_saved >= limit:
            break
        
    ##########################################################################  
                 
    time_end = time.time()
    run_time = int(time_end-time_start)
    print('----------------------------------')
    print(str(counter_saved)+' shot(s) saved to file')
    print('Total run time: '+str(run_time) + ' seconds')
    
    #Write shotnoloc file
    data_shotno_row_index_df = pd.DataFrame(data_shotno_row_index)
    data_shotno_row_index_df.to_csv(PATH_SHOTNOLOC, header=False, index=False)   
    print('Wrote shotnoloc file')
    
    #Write summary file
    file_summary = open(PATH_SUMMARY, 'w')
    file_summary.write('File name: '+ filename+'\n')
    file_summary.write('Date: ' + time.ctime()+'\n')
    file_summary.write('Shotno range: '+str(shotno_start)+'-'+str(shotno_end)+'\n')
    file_summary.write('Number of shots saved: ' + str(counter_saved)+'\n')
    file_summary.write('Downsampling rate: ' + str(downsampling)+'\n')
    file_summary.write('Shot style filtering rule: ' + str(filtering)+'\n')
    file_summary.write('Signals: '+str(signals)+'\n')
    file_summary.write('Total run time: ' + str(run_time)+' seconds\n')
    file_summary.close()
    print('Wrote summary file')   
    
    return 

if __name__ == "__main__":
    main()