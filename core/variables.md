# Variables

Table with the dimensions of the variables used in the MPC.

Variable                Distributed             Centralized
----------------------------------------------------------------
self.x                  (K+1, n_x)              (K+1, n_x, m)
self.u                  (K, n_u)                (K, n_u, m)
self.x_0                (n_x, )                 (n_x, m)
self.x_goal             (n_x, )                 (n_x, m)
self.x_pred             (K+1, n_x, m-1)         None
self.x_prev             (K+1, n_x)              None
self.w_curr             (m-1, )                 (m, m)
self.w_target           (m-1, )                 (m, m)
self.alpha_u            float                   float
self.alpha_g            float                   float
self.alpha_w            (m-1,)                  (m, m)
self.u_min              (n_u, )                 (n_u, )
self.u_max              (n_u, )                 (n_u, )
self.u_rate_min         (n_u, )                 (n_u, )
self.u_rate_max         (n_u, )                 (n_u, )
self.u_tot_max          float                   float
self.x_min              (n_x, )                 (n_x, )
self.x_max              (n_x, )                 (n_x, )
self.d_min              float                   float

Note: in the centralized case, the diagonal of the alpha_w, w_curr, and w_target matrices is 0, and those matrices are symmetric.