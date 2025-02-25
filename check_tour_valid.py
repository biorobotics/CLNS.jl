from os.path import expanduser
import numpy as np
import pickle

class PWLTrj(object):
  def __init__(self, tw_p0, tw_vels, tws):
    self.tw_p0 = tw_p0
    self.tw_vels = tw_vels
    self.tws = tws

  def __call__(self, t):
    for tw, tw_p0, tw_vel in zip(self.tws, self.tw_p0, self.tw_vels):
      if tw[0] <= t <= tw[1]:
        return tw_p0 + tw_vel*t

    assert(False)

instance_path = expanduser("~/catkin_ws/src/mapf/data/02_24_2025/mt_tsp_clns_test_instances/targ200_win108_random_seed0_occprob0.0_vmaxa5.0_2win_rad0/")

tw_p0 = np.load(instance_path + '/tw_p0.npy')
tw_vels = np.load(instance_path + '/tw_vels.npy')
tws = np.load(instance_path + '/tws.npy')
tw_to_target_ptr = np.load(instance_path + '/tw_to_target_ptr.npy')
with open(instance_path + '/target_to_tw_ptr.pkl', 'rb') as f:
  target_to_tw_ptr = pickle.load(f)
start_pos = np.load(instance_path + '/depot_pos.npy')
vmax_agent = np.load(instance_path + '/vmax_agent.npy')

pos_trj_per_target = [PWLTrj(tw_p0[ptr], tw_vels[ptr], tws[ptr]) for ptr in target_to_tw_ptr]

with open('validity_check_200_targets.tour', 'r') as f:
  lines = f.readlines()

for line in lines:
  if 'Tour Cost' in line:
    cost = float(line[line.find(':') + 2:-1])
  if 'Tour' in line and 'Tour Cost' not in line and 'Tour History' not in line:
    pt_seq = np.zeros((len(target_to_tw_ptr) + 1, 3))
    start_idx = line.find('[') + 1
    end_idx = line.find(']')
    for pt_idx, pt_str in enumerate(line[start_idx:end_idx].split(';')):
      coords = [float(n) for n in pt_str.split(' ') if len(n)]
      pt_seq[pt_idx] = np.array(coords)

  if 'Set sequence' in line:
    start_idx = line.find('[') + 1
    end_idx = line.find(']')
    target_seq = [int(n) - 2 for n in line[start_idx:end_idx].split(',')]
    print(target_seq)

for target_idx, pt in zip(target_seq, pt_seq):
  t = pt[0]
  if target_idx != -1:
    assert(np.all(np.abs(pos_trj_per_target[target_idx](t) - pt[1:]) < 1e-10))

check_cost = 0.
for seq_idx, (pt1, pt2) in enumerate(zip(pt_seq[:-1], pt_seq[1:])):
  if target_seq[seq_idx + 1] != -1:
    check_cost += np.linalg.norm(pt2[1:] - pt1[1:])
    assert(np.linalg.norm(pt2[1:] - pt1[1:]) <= vmax_agent*(pt2[0] - pt1[0]))
if target_seq[0] != -1:
  check_cost += np.linalg.norm(pt_seq[0, 1:] - pt_seq[-1, 1:])
  assert(np.linalg.norm(pt_seq[0, 1:] - pt_seq[-1, 1:]) <= vmax_agent*(pt_seq[0, 0] - pt_seq[-1, 0]))
assert(abs(check_cost - cost) < 1e-10)
