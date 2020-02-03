
void __device__ apply_mic
(
    real box[6], real *x12, real *y12, real *z12
)
{
    if      (*x12 < - box[3]) { *x12 += box[0]; } 
    else if (*x12 > + box[3]) { *x12 -= box[0]; }
    if      (*y12 < - box[4]) { *y12 += box[1]; } 
    else if (*y12 > + box[4]) { *y12 -= box[1]; }
    if      (*z12 < - box[5]) { *z12 += box[2]; } 
    else if (*z12 > + box[5]) { *z12 -= box[2]; }
}

void __global__ gpu_find_neighbor
(
    int N, int MN, int *g_NN, int *g_NL, real box[6], 
    real *g_x, real *g_y, real *g_z, real cutoff2
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        int count = 0;
        real x1 = g_x[n1];
        real y1 = g_y[n1];
        real z1 = g_z[n1];
        for (int n2 = 0; n2 < N; n2++)
        {
            real x12 = g_x[n2] - x1;
            real y12 = g_y[n2] - y1;
            real z12 = g_z[n2] - z1;
            apply_mic(box, &x12, &y12, &z12);
            real d12_square = x12*x12 + y12*y12 + z12*z12;
            if ((n2 != n1) && (d12_square < cutoff2))
            {
                g_NL[count++ * N + n1] = n2;
            }
        }
        g_NN[n1] = count;
    }
}

void __global__ gpu_find_force
(
    real lj[5], int N, int *g_NN, int *g_NL, real box[6],
    real *g_x, real *g_y, real *g_z,
    real *g_fx, real *g_fy, real *g_fz, real *g_pe
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        real fx = 0.0;
        real fy = 0.0;
        real fz = 0.0;
        real potential = 0.0;
        int NN = g_NN[i];
        real x_i = g_x[i];
        real y_i = g_y[i];
        real z_i = g_z[i];

        for (int k = 0; k < NN; ++k)
        {   
            int j = g_NL[i + N * k];
            real x_ij  = g_x[j] - x_i;
            real y_ij  = g_y[j] - y_i;
            real z_ij  = g_z[j] - z_i;
            apply_mic(box, &x_ij, &y_ij, &z_ij);
            real r2 = x_ij*x_ij + y_ij*y_ij + z_ij*z_ij;
            if (r2 > lj[0]) { continue; }
            real r2inv = 1.0 / r2;
            real r4inv = r2inv * r2inv;
            real r6inv = r2inv * r4inv;
            real r8inv = r4inv * r4inv;
            real r12inv = r4inv * r8inv;
            real r14inv = r6inv * r8inv;
            real f_ij = lj[1] * r8inv - lj[2] * r14inv;
            potential += lj[4] * r12inv - lj[3] * r6inv;
            fx += f_ij * x_ij;
            fy += f_ij * y_ij;
            fz += f_ij * z_ij;
        }
        g_fx[i] = fx; 
        g_fy[i] = fy; 
        g_fz[i] = fz; 
        g_pe[i] = potential * 0.5;
    }
}


void __global__ gpu_integrate
(
    int N, real time_step, real time_step_half,
    real *g_m, real *g_x, real *g_y, real *g_z,
    real *g_vx, real *g_vy, real *g_vz,
    real *g_fx, real *g_fy, real *g_fz, 
    real *g_ke, int flag
)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        real mass = g_m[n];
        real mass_inv = 1.0 / mass;
        real ax = g_fx[n] * mass_inv;
        real ay = g_fy[n] * mass_inv;
        real az = g_fz[n] * mass_inv;
        real vx = g_vx[n];
        real vy = g_vy[n];
        real vz = g_vz[n];
        vx += ax * time_step_half;
        vy += ay * time_step_half;
        vz += az * time_step_half;
        g_vx[n] = vx;
        g_vy[n] = vy;
        g_vz[n] = vz;

        if (flag == 1) 
        { 
            g_x[n] += vx * time_step; 
            g_y[n] += vy * time_step; 
            g_z[n] += vz * time_step; 
        }
        else
        {
            g_ke[n] = (vx*vx + vy*vy + vz*vz) * mass * 0.5;
        }
    }
}

void __global__ gpu_scale_velocity
(
    int N, real scale_factor, 
    real *g_vx, real *g_vy, real *g_vz
)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    { 
        g_vx[n] *= scale_factor;
        g_vy[n] *= scale_factor;
        g_vz[n] *= scale_factor;
    }
}