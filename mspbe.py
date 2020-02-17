import numpy as np
import time
import torch

def calc_mspbe_torch(exp, rho):
    A_theta_minus_b = torch.mv(exp.A,exp.theta) - exp.b
    return (1/2)*torch.dot(A_theta_minus_b, torch.mv(exp.C_inv, A_theta_minus_b)) + torch.mul(rho/2, torch.dot(exp.theta, exp.theta)) - torch.mul(exp.rho_omega/2, torch.dot(exp.omega, exp.omega))

def mspbe_grad_theta(theta, omega, A, rho):
    #return -torch.mv(torch.t(A_t), omega)s
    return theta*rho - torch.mv(torch.t(A), omega)

def mspbe_grad_omega(theta, omega, A, b, C, rho_omega):
    return torch.mv(A, theta) - b + torch.mv(C, omega) - omega*rho_omega

def get_stoc_abc_torch(exp,t_j):
    trans_data = exp.trans_data[t_j,:]
    s_t = int(trans_data[0])
    s_t_1 = int(trans_data[3])
    r_t = trans_data[2]
    phi_s_t = exp.Phi[s_t, :]
    phi_s_t_1 = exp.Phi[s_t_1, :]
    A_t = torch.ger(phi_s_t, phi_s_t-exp.gamma*phi_s_t_1) + torch.mul(torch.eye(exp.nFeatures, dtype=torch.float64, device=exp.device), exp.rho_ac)
    b_t = r_t * phi_s_t
    C_t = torch.ger(phi_s_t, phi_s_t) + torch.mul(torch.eye(exp.nFeatures, dtype=torch.float64, device=exp.device), exp.rho_ac)
    return A_t, b_t, C_t

def calc_eig_max_AtCinvA(exp):
    return torch.symeig(torch.matmul(torch.t(exp.A), torch.matmul(exp.C_inv, exp.A)))[0][exp.nFeatures - 1]

def get_eig_max_C(exp):
    return torch.symeig(exp.C)[0][exp.nFeatures-1]

def calc_cond_C(exp, eig_max_C):
    eig_min_C = torch.symeig(exp.C)[0][0]
    return eig_max_C/eig_min_C



# class mspbe_graph:
#     def __init__(self, nFeatures):
#         self.g, self.mspbe, self.A_ph, self.b_ph, self.C_inv_ph, self.theta_ph = self.build_mspbe_graph(nFeatures)
#         self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(log_device_placement=True))
#
#     def build_mspbe_graph(self, nFeatures):
#         g = tf.Graph()
#         with g.as_default():
#             tf_scalar = tf.constant(0.5, dtype=tf.float64)
#             A_ph = tf.placeholder(tf.float64,shape=(nFeatures,nFeatures))
#             b_ph = tf.placeholder(tf.float64,shape=(nFeatures))
#             C_inv_ph = tf.placeholder(tf.float64,shape=(nFeatures,nFeatures))
#             theta_ph = tf.placeholder(tf.float64,shape=(nFeatures))
#
#             Atheta_minus_b = tf.subtract(tf.tensordot(A_ph, theta_ph, axes=[[1], [0]]), b_ph)
#             C_inv_Atheta_minus_b = tf.tensordot(C_inv_ph, Atheta_minus_b, axes=[[1], [0]])
#             mspbe = tf.scalar_mul(tf_scalar, tf.tensordot(Atheta_minus_b, C_inv_Atheta_minus_b, 1))
#         return g, mspbe, A_ph, b_ph, C_inv_ph, theta_ph

# class mspbe_stoc_grad_theta_graph:
#     def __init__(self, nFeatures):
#         self.g, self.stoc_grad_theta, self.A_t_ph, self.omega_ph = self.build_mspbe_stoc_grad_theta_graph(nFeatures)
#         self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(log_device_placement=True))
#
#     def build_mspbe_stoc_grad_theta_graph(self, nFeatures):
#         g = tf.Graph()
#         with g.as_default():
#             A_t_ph = tf.placeholder(tf.float64,shape=(nFeatures,nFeatures))
#             omega_ph = tf.placeholder(tf.float64,shape=(nFeatures))
#             stoc_grad_theta = tf.negative(tf.tensordot(tf.transpose(A_t_ph), omega_ph, axes=[[1], [0]]))
#         return g, stoc_grad_theta, A_t_ph, omega_ph
#
# class mspbe_stoc_grad_omega_graph:
#     def __init__(self, nFeatures):
#         self.g, self.stoc_grad_omega, self.A_t_ph, self.b_t_ph, self.C_t_ph, self.theta_ph, self.omega_ph = self.build_mspbe_stoc_grad_omega_graph(nFeatures)
#         self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(log_device_placement=True))
#
#     def build_mspbe_stoc_grad_omega_graph(self, nFeatures):
#         g = tf.Graph()
#         with g.as_default():
#             A_t_ph = tf.placeholder(tf.float64,shape=(nFeatures,nFeatures))
#             b_t_ph = tf.placeholder(tf.float64,shape=(nFeatures))
#             C_t_ph = tf.placeholder(tf.float64,shape=(nFeatures,nFeatures))
#             theta_ph = tf.placeholder(tf.float64,shape=(nFeatures))
#             omega_ph = tf.placeholder(tf.float64,shape=(nFeatures))
#
#             A_t_theta_minus_b_t = tf.subtract(tf.tensordot(A_t_ph, theta_ph, axes=[[1], [0]]), b_t_ph)
#             C_t_omega = tf.tensordot(C_t_ph, omega_ph, axes=[[1], [0]])
#             stoc_grad_omega = tf.add(A_t_theta_minus_b_t, C_t_omega)
#         return g, stoc_grad_omega, A_t_ph, b_t_ph, C_t_ph, theta_ph, omega_ph
#
# class outer_product_graph:
#     def __init__(self, nFeatures):
#         self.g, self.outer_prod, self.a_ph, self.b_ph = self.build_tf_outer_prod_graph(nFeatures)
#         self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(log_device_placement=True))
#
#     def build_tf_outer_prod_graph(self, nFeatures):
#         g = tf.Graph()
#         with g.as_default():
#             a_ph = tf.placeholder(tf.float64,shape=(nFeatures))
#             b_ph = tf.placeholder(tf.float64,shape=(nFeatures))
#             outer_prod = tf.tensordot(tf.expand_dims(a_ph,0), tf.expand_dims(b_ph,0), axes=[[0],[0]])
#         return g, outer_prod, a_ph, b_ph

# def get_stoc_abc(exp,t_j):
#     mdp = exp.mdp
#     trans_data = mdp.trans_data[t_j, :]
#     s_t = int(trans_data[0])
#     s_t_1 = int(trans_data[3])
#     r_t = trans_data[2]
#     phi_s_t = mdp.Phi[s_t, :]
#     phi_s_t_1 = mdp.Phi[s_t_1, :]
#
#     start = time.time()
#     if exp.use_gpu:
#         graph = exp.outer_product_graph
#         A_t = graph.sess.run(graph.outer_prod, feed_dict={graph.a_ph:phi_s_t, graph.b_ph:phi_s_t-mdp.gamma*phi_s_t_1})
#         C_t = graph.sess.run(graph.outer_prod, feed_dict={graph.a_ph:phi_s_t, graph.b_ph:phi_s_t})
#     else:
#         A_t = np.outer(phi_s_t, phi_s_t - mdp.gamma * phi_s_t_1)
#         C_t = np.outer(phi_s_t, phi_s_t)
#     b_t = r_t * phi_s_t
#     end = time.time()
#     #exp.tot_comp_time += end - start
#
#     return A_t, b_t, C_t

# def mspbe_batch_grads(theta,omega,batch_A,batch_b,batch_C):
#     return -batch_A.T@omega, batch_A@theta - batch_b + batch_C@omega

# def obj_primal(theta,omega, A_t):
#     return -torch.dot(omega, torch.mv(A_t, theta))
#
# def obj_dual(theta, omega, A_t, b_t, C_t):
#     return -torch.dot(omega, torch.mv(A_t, theta)) - 0.5*torch.dot(omega, torch.mv(C_t, omega)) + torch.dot(omega, b_t)

# def calc_mspbe(exp):
#     #a = (rho/2)*np.dot(theta,theta) - np.dot(omega,A@theta) - (1/2)*np.dot(omega,C@omega) + np.dot(omega,b)
#     start = time.time()
#     if exp.use_gpu:
#         res = exp.mspbe_graph.sess.run(exp.mspbe_graph.mspbe, feed_dict={exp.mspbe_graph.A_ph:exp.mdp.A, exp.mspbe_graph.b_ph:exp.mdp.b, exp.mspbe_graph.C_inv_ph:exp.mdp.C_inv, exp.mspbe_graph.theta_ph:exp.theta})
#     else:
#         res = (1/2)*np.dot(exp.mdp.A@exp.theta-exp.mdp.b, exp.mdp.C_inv@(exp.mdp.A@exp.theta-exp.mdp.b))
#     end = time.time()
#     exp.tot_comp_time += end-start
#     return res

# def mspbe_grads(theta,omega,mdp):
#     return mdp.rho*theta - mdp.A.T@omega, mdp.A@theta - mdp.b + mdp.C@omega

# def mspbe_stoc_grad_omega(exp, A_t, b_t, C_t):
#     start = time.time()
#     if exp.use_gpu:
#         graph = exp.stoc_grad_omega_graph
#         res = graph.sess.run(graph.stoc_grad_omega, feed_dict={graph.A_t_ph:A_t, graph.b_t_ph:b_t, graph.C_t_ph:C_t, graph.theta_ph:exp.theta, graph.omega_ph:exp.omega})
#     else:
#         res = A_t@exp.theta - b_t + C_t@exp.omega
#     end = time.time()
#     exp.tot_comp_time += end - start
#     return res

# def mspbe_stoc_grad_theta(exp, A_t):
#     start = time.time()
#     if exp.use_gpu:
#         graph = exp.stoc_grad_theta_graph
#         res = graph.sess.run(graph.stoc_grad_theta, feed_dict={graph.A_t_ph:A_t, graph.omega_ph:exp.omega})
#     else:
#         res = -A_t.T@exp.omega
#     end = time.time()
#     exp.tot_comp_time += end - start
#     return res

# def mspbe_grads_torch(theta,omega,mdp,rho):
#     return rho*theta-torch.mv(torch.t(mdp.A), omega), torch.mv(mdp.A, theta) - mdp.b + torch.mv(mdp.C,omega)

# def mspbe_batch_grads_torch(theta,omega,batch_A,batch_b,batch_C,rho):
#     return rho*theta-torch.mv(torch.t(batch_A), omega), torch.mv(batch_A, theta) - batch_b + torch.mv(batch_C, omega)