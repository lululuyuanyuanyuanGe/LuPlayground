# N: Number of Grid Intervals (e.g., 8)
# R: Range for Grids (e.g., [0, pi])
# P: Probability of Each Grids
# G: Adaptive Action Grids with Embedding of Size E
# cdf: Cumulative Distribution Function
# ppf: Percent Point Function
# create adptive action grids from gaussian distributions
for gaussians, grid_params in GS(theta, phi, r, roll, pitch, yaw):
for (mu, sigma), (R, N) in gaussians, grid_params:
P = linspace(cdf(R, mu, sigma), cdf(R, mu, sigma), N + 1)
G.x = ppf(P, mu, sigma) # coordinates
G.fea = Embedding(N, E) # features
G.add_gripper() # add gripper 2 grids
# linearization 3d grids to share parameters with llm embedding
# trans: [N_theta * N_phi * N_r], rot: [N_roll * N_pitch * N_yaw]
# gripper: [N_gripper]
G.linearization()
# T: Number of Timesteps
# D: Dataset
for t in range(0, T):
# if encode
a = D(t) # normalized action [theta, phi, r, roll, pitch, yaw, gripper]
# digitize continuous actions to 3d grids
d_theta, d_phi, d_r = digitize(G, theta, phi, r) # trans
d_roll, d_pitch, d_yaw = digitize(G, roll, pitch, yaw) # rot
# linearization
id_trans = linearize(d_theta, d_phi, d_r)
id_rot = linearize(d_roll, d_pitch, d_yaw)
id_gripper = 1 if gripper > 0.5 else 0 # gripper
token_trans, token_rot, token_gripper = G.fea(id_trans, id_rot, id_gripper)
# if decode
(id_trans, id_rot, id_gripper) = SpatialVLA([image], prompt) # predict 3 action token id
d_theta, d_phi, d_r = gridification(G, id_trans)
d_roll, d_pitch, d_yaw = gridification(G, id_rot)
gripper = id_gripper
a = unnomalize(d_theta, d_phi, d_r, d_roll, d_pitch, d_yaw, gripper)