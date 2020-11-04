from signal.signal import Signal

def main():
    # define object
    myobject = Signal('\nztpi\n')
    # print object name
    print(myobject)
    #import data
    myobject.load_data('./ztpi_data.txt')
    #check key parameters of sampling features of the input signal from in-situ device
    print(f'mean, median, and standard deviation of input sampling rate: \n{myobject.compute_sampling_spacing()}\n')
    #check key parameters of period features of the structural reposnse
    print(f'mean, median, and standard deviation of signal main period: \n{myobject.compute_period()}\n')

    #return LSF object cost function
    print(f'{myobject.lsq_resutls().cost}')
    # return Soft_L1 fit object
    print(f'{myobject.softL1_results()}')
    # return Huber fit object's solve vector - first element
    print(f'{myobject.huber_results().x[0]}')
    myobject.compute_period()

if __name__ == '__main__':
    main()


    

