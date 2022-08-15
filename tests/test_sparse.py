import python_cpp_skel as sk
import numpy as np 

def test_add():
    assert sk.add(1, 2) == 3

def test_subtract():
    assert sk.subtract(1, 2) == -1

Adf = np.array(
        [[1,2,1],
        [2,1,0],
        [-1,1,2]], 
        order="F", dtype='float64')

Adc = np.array(
        [[1,2,1],
        [2,1,0],
        [-1,1,2]], 
        order="C", dtype='float64')

# simple mutate
sk.mutate(Adf)

sk.incr_matrix(Adf, 100000.0)
# sk.incr_matrix(Adc, 100000.0) # doesn't work because of stride
sk.incr_matrix_any(Adf, 100000.0)
sk.incr_matrix_any(Adc, 100000.0)

## works 
Vdf = np.array(
        [1,2,1],
        order="F", dtype='float64')

Vdc = np.array(
        [1,2,1],
        order="C", dtype='float64')

sk.incr_vector(Vdc, 3.0) # works because stride doesn't matter
sk.incr_vector(Vdf, 5.0)
# sk.incr_vector_any(Vdc, 3.0)
# sk.incr_vector_any(Vdf, 3.0)

## snippets

## Aff = np.array(
##         [[1,2,1],
##         [2,1,0],
##         [-1,1,2]], 
##         order="F", dtype='float32')
## Afc = np.array(
##         [[1,2,1],
##         [2,1,0],
##         [-1,1,2]], 
##         order="C", dtype='float32')

## Vff = np.array(
##         [1,2,1],
##         order="F", dtype='float32')
## Vfc = np.array(
##         [1,2,1],
##         order="C", dtype='float32')
## 
## 
## sk.sparsify_r(Af)
## sk.sparsify_c(Af)
## sk.sparsify_r(Ad)
## sk.sparsify_c(Ad)

# print(sk.sparsify_r(Af).todense())
# print(sk.sparsify_c(Af).todense())
# 
# print(sk.sparsify_r(Ad).todense())
# print(sk.sparsify_c(Ad).todense())

sk.sparsify_r(Adf).todense()