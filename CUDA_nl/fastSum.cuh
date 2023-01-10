__global__ void prescan_arbitrary(uint32_t* output, uint32_t* input, uint32_t n, uint32_t powerOfTwo, uint32_t l);
__global__ void prescan_large(uint32_t* output, uint32_t* input, uint32_t n, uint32_t* sums, uint32_t l);
__global__ void add(uint32_t* output, uint32_t length, uint32_t* n);
__global__ void add(uint32_t* output, uint32_t length, uint32_t* n1, uint32_t* n2);
