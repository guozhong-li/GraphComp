import numpy as np
import os
import ctypes

def cartesian_product(arrays):
    la = len(arrays)
    dtype = arrays[0].dtype
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def my_compress(data_cpy, preds_cpy, eb, filename):
    lib_path = '/path/to/the/library/of/lib_sz.so'
    clibrary = ctypes.CDLL(lib_path, use_last_error=True)
    ''' 
    void compress(float* data, 
                 float* preds, 
                 double eb, 
                 float data_range, 
                 size_t num_elements, 
                 float mean, 
                 float std,
                 )
    '''
    compress = clibrary.compress
    compress.argtypes = [ctypes.POINTER(ctypes.c_float), 
                         ctypes.POINTER(ctypes.c_float), 
                         ctypes.c_double, 
                         ctypes.c_float, 
                         ctypes.c_size_t, 
                         ctypes.c_float, 
                         ctypes.c_float, 
                         ctypes.POINTER(ctypes.c_char),
                         ctypes.POINTER(ctypes.c_int),
                        ]
    compress.restype = ctypes.c_void_p
    
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    
    data_size = len(data_cpy)
    data_range = np.ptp(data_cpy)
    mean = np.mean(data_cpy)
    std = np.std(data_cpy)
    quant_hold = np.empty_like(data_cpy, dtype=np.int32)
    
    compress(data_cpy.flatten().ctypes.data_as(c_float_p), #data
          preds_cpy.ctypes.data_as(c_float_p), #preds
          eb, #error bound
          data_range, #val_range
          data_size, #num elements
          mean, #data mean
          std, #data std
          bytes(filename, encoding='utf8'), 
          quant_hold.ctypes.data_as(c_int_p),
         )
    cmp_size = os.path.getsize(filename)
    # print(f'CR = {cmp_size}')
    return cmp_size, quant_hold

def decompress(filename, data, preds, eb):
    lib_path = '/path/to/the/library/of/lib_sz.so/lib_sz.so'
    clibrary = ctypes.CDLL(lib_path, use_last_error=True)
    ''' 
    void decompress(const char* cmpPath,
                    size_t num_elements,
                    )
    '''
    decompress = clibrary.decompress
    decompress.argtypes = [ctypes.c_char_p, 
                           ctypes.POINTER(ctypes.c_float), 
                           ctypes.c_size_t, 
                           ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_int)]
    decompress.restype = ctypes.c_void_p
    
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    
    res_hold = np.zeros_like(data)
    quant_hold = np.empty_like(data, dtype=np.int32)
    
    decompress(bytes(filename, encoding='utf8'),
         preds.ctypes.data_as(c_float_p),
         len(data),
         res_hold.ctypes.data_as(c_float_p),
         quant_hold.ctypes.data_as(c_int_p)
        )
    
    dec_out = res_hold.copy()
    max_err = np.max(np.fabs(data - dec_out))
    print(max_err)
    data_range = np.ptp(data)
    print(round(max_err, 4) - round(eb*data_range, 4))
    assert np.allclose(max_err, eb*data_range, atol=1e-4), f'{filename}: {max_err, eb*data_range}'
    return dec_out

def main():
    raw_data = np.fromfile('/path/to/original/era5/data/era5_2018.dat', dtype=np.float64).reshape(-1, 721, 1440)
    print('raw_data,shape()', raw_data.shape)
    data = np.load('/path/to/the/restructed/data/by/g2g/re_era5_s25s1m1_wostd5_t1seg.npy').reshape(-1, 721, 1440)
    ref_data = data
    print('ref_data,shape()', ref_data.shape)

    print("error bound:  1e-2")
    
    cm_size, qout = my_compress(data_cpy=raw_data.copy().flatten(),
                            preds_cpy=ref_data.copy().flatten(),
                            eb=1e-2,
                            filename=f'./era5-1e-2.sz')
    print((ref_data.size*ref_data.itemsize)/(cm_size))
    print(ref_data.size, ref_data.itemsize, cm_size)

    print("error bound:  1e-3")
    cm_size, qout = my_compress(data_cpy=raw_data.copy().flatten(),
                            preds_cpy=ref_data.copy().flatten(),
                            eb=1e-3,
                            filename=f'./era5-1e-3.sz')
    print((ref_data.size*ref_data.itemsize)/(cm_size))
    print(ref_data.size, ref_data.itemsize, cm_size)
    
    print("error bound:  1e-4")
    cm_size, qout = my_compress(data_cpy=raw_data.copy().flatten(),
                            preds_cpy=ref_data.copy().flatten(),
                            eb=1e-4,
                            filename=f'./era5-1e-4.sz')
    print((ref_data.size*ref_data.itemsize)/(cm_size))
    print(ref_data.size, ref_data.itemsize, cm_size)


if __name__ == "__main__":
    
    main()