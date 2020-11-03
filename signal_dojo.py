from signal.signal import Signal

if __name__ == '__main__':
    # define object
    myobject = Signal('\nztpi\n')
    # print object name
    print(myobject)
    #import data
    myobject.load_data('./ztpi_data.txt')
    #check key parameters of sampling features of the input signal from in-situ device
    print(f'mean, median, and standard deviation of input sampling rate: \n{myobject.compute_sampling_spacing()}\n')
    #return LSF object cost function
    print(f'{myobject.lsq_resutls().cost}')
    # return Soft_L1 fit object
    print(f'{myobject.softL1_results()}')
    # return Huber fit object's solve vector - first element
    print(f'{myobject.huber_results().x[0]}')

    

