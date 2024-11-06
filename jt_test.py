#This example shows how to use multi dimension data with CUDA.
import jittor as jt
from jittor import Function
jt.flags.use_cuda = 1

class Func(Function):
    def execute(self, a, b,c):
        self.save_vars = a, b,c

        d=jt.Var([a.shape[0]])
        # c = 10.
        output = jt.full(a.shape, 0, dtype=jt.float32)
        output = jt.code(a.shape, a.dtype, [a,b,c,d],
            cuda_src='''
                __global__ static void kernel1(@ARGS_DEF) {
                    @PRECALC
                    for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
                    for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)
                        @out(i,j) = @in0(i,j)*@in1(i,j) +@in2(0) ;
                }
                kernel1<<<32, 32>>>(@ARGS);
            ''')

        return output

    # def grad(self, grad):
    #     a, b = self.save_vars
    #     return jt.code([a.shape, b.shape], [a.dtype, b.dtype], [a, b, grad],
    #         cuda_src='''
    #             __global__ static void kernel2(@ARGS_DEF) {
    #                 @PRECALC
    #                 for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
    #                 for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x) {
    #                     @out0(i,j) = @in2(i,j)*@in1(i,j);
    #                     @out1(i,j) = @in2(i,j)*@in0(i,j);
    #                 }
    #             }
    #             kernel2<<<32, 32>>>(@ARGS);
    #         ''')

a = jt.random((2,2))
b = jt.random((2,2))
c = jt.Var((10))
func = Func()
d = func(a,b,c)
print('a : \n',a)
print('b : \n',b)
print('c : \n',c)
print('c : \n',d)
# print(jt.grad(c, [a, b]))