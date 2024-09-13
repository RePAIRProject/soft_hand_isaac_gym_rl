import torch
import numpy as np


class PySoftHandMimicPlugin():
    
    def __init__(self, num_envs, asset_handle, sim_handle, find_joint_index_func, sim_time_func, device='cuda:0'):
        self.num_envs = num_envs
        self.asset_handle = asset_handle
        self.sim_handle = sim_handle
        self.find_asset_joint_index = find_joint_index_func
        self.get_sim_time =  sim_time_func
        self.ns_name = "right_hand_v1_2_research"
        finger_names = ["thumb", "index", "middle", "ring", "little"]
        joint_names = ["knuckle", "proximal", "middle", "distal"]
        self.old_time = 0.0
        self.device = device

        fingers_dict = {'thumb': {}, 'index': {}, 'middle': {}, 'ring': {}, 'little': {}}
        self.joints_dim = [3, 4, 4, 4, 4]
        
        # commons joints             
        self.fingers_joint     = [0]*5 
        self.q_fingers         = torch.zeros([self.num_envs, 5, 4], device=self.device) if device == 'cuda:0' else np.zeros([self.num_envs, 5, 4])
        self.dq_fingers        = torch.zeros([self.num_envs, 5, 4], device=self.device) if device == 'cuda:0' else np.zeros([self.num_envs, 5, 4]) 
        # mimic joints
        self.fingers_mimic_joint = [0]*5           
        self.qMimic_fingers      = torch.zeros([self.num_envs, 5, 3], device=self.device) if device == 'cuda:0' else np.zeros([self.num_envs, 5, 3])
        self.dqMimic_fingers     = torch.zeros([self.num_envs, 5, 3], device=self.device) if device == 'cuda:0' else np.zeros([self.num_envs, 5, 3])

        for i, dim in enumerate(self.joints_dim):
            self.fingers_joint[i]          = [0] * dim      
            self.fingers_mimic_joint[i]    = [0] * (dim - 1) 
       

        finger_idx = 0
        for finger in self.fingers_joint:
            joint_basic_idx = 0
            for joint in finger:
                if finger_idx == 0 and joint_basic_idx == 2:
                    temp_name_basic = self.ns_name + '_' + finger_names[finger_idx] + '_' + joint_names[joint_basic_idx+1] + "_joint"
                else:
                    temp_name_basic = self.ns_name + '_' + finger_names[finger_idx] + '_' + joint_names[joint_basic_idx] + "_joint"
                self.fingers_joint[finger_idx][joint_basic_idx] = self.find_asset_joint_index(asset_handle, temp_name_basic)
                print("Joints number ", str(self.fingers_joint[finger_idx][joint_basic_idx]), "name = ", temp_name_basic, "\n")
                joint_basic_idx += 1
            
            joint_mimic_idx = 0
            for joint in finger[:-1]:
                if finger_idx == 0 and joint_mimic_idx == 1:
                    temp_name_mimic = self.ns_name + '_' + finger_names[finger_idx] + '_' + joint_names[joint_mimic_idx+2] + "_virtual_joint"
                else:
                    temp_name_mimic = self.ns_name + '_' + finger_names[finger_idx] + '_' + joint_names[joint_mimic_idx+1] + "_virtual_joint"
                self.fingers_mimic_joint[finger_idx][joint_mimic_idx] = self.find_asset_joint_index(asset_handle, temp_name_mimic)
                print("Joints number ", str(self.fingers_mimic_joint[finger_idx][joint_mimic_idx]), "name = ", temp_name_mimic, "\n")
                joint_mimic_idx += 1
            finger_idx += 1

        self.init_constants()


    def get_fingers_state(self, current_positions, current_velocities):
        finger_idx = 0
        for finger in self.fingers_joint:
            joint_basic_idx = 0
            for joint in finger:
                self.q_fingers[:, finger_idx, joint_basic_idx] = current_positions[:, self.fingers_joint[finger_idx][joint_basic_idx]]
                self.dq_fingers[:, finger_idx, joint_basic_idx] = current_velocities[:, self.fingers_joint[finger_idx][joint_basic_idx]]
                joint_basic_idx += 1
            finger_idx += 1
        
        finger_idx = 0
        for finger in self.fingers_mimic_joint:
            joint_mimic_idx = 0
            for joint in finger:
                self.qMimic_fingers[:, finger_idx, joint_mimic_idx] = current_positions[:, self.fingers_mimic_joint[finger_idx][joint_mimic_idx]]
                self.dqMimic_fingers[:, finger_idx, joint_mimic_idx] = current_velocities[:, self.fingers_mimic_joint[finger_idx][joint_mimic_idx]]
                joint_mimic_idx += 1
            finger_idx += 1

    def OnUpdateSoftSyn(self, ref, current_positions, current_velocities):
        # retrieve simulation time
        t = self.get_sim_time(self.sim_handle)

        self.get_fingers_state(current_positions, current_velocities)

        tau_coupling = torch.zeros([self.num_envs, 5, 3], device=self.device) if self.device == 'cuda:0' else np.zeros([self.num_envs, 5, 3]) # np.array([[[0]*3]*5]*self.num_envs)
        for i in range(5):
            for j in range(3):
                tau_coupling[:, i, j] = self.spring_k_mimic_each[i][j] * (self.q_fingers[:, i, j+1] - self.qMimic_fingers[:, i, j])

    

        # tau = torch.zeros([self.num_envs, 39], device=self.device) if self.device == 'cuda:0' else np.zeros([self.num_envs, 39])
        tau = torch.zeros([self.num_envs, 40], device=self.device) if self.device == 'cuda:0' else np.zeros([self.num_envs, 40])
        tau_el = torch.zeros([self.num_envs, 5, 4], device=self.device) if self.device == 'cuda:0' else np.zeros([self.num_envs, 5, 4])
        tau_el_mimic = torch.zeros([self.num_envs, 5, 4], device=self.device) if self.device == 'cuda:0' else np.zeros([self.num_envs, 5, 3])

        finger_idx = 0
        # for finger in self.q_fingers[0]:
        for finger in self.fingers_joint:
            joint_idx = 0
            for joint in finger:
                if joint_idx == 0:
                    tau_el[:, finger_idx, joint_idx] = self.spring_k_each[finger_idx][joint_idx]*(self.fingers_syn_S[finger_idx][joint_idx]*ref - self.q_fingers[:, finger_idx, joint_idx])
                else:
                    tau_el[:, finger_idx, joint_idx] = self.spring_k_each[finger_idx][joint_idx]*(self.fingers_syn_S[finger_idx][joint_idx]*ref - self.q_fingers[:, finger_idx, joint_idx]) - tau_coupling[:, finger_idx, joint_idx-1]  
                tau[:, self.fingers_joint[finger_idx][joint_idx]] = tau_el[:, finger_idx, joint_idx]
                joint_idx += 1
            finger_idx += 1

        finger_idx = 0
        # for finger in self.qMimic_fingers[0]:
        for finger in self.fingers_mimic_joint:
            joint_idx = 0
            for joint in finger:
                tau_el_mimic[:, finger_idx, joint_idx] = self.spring_k_each[finger_idx][joint_idx]*(self.fingers_syn_S[finger_idx][joint_idx]*ref - self.qMimic_fingers[:, finger_idx, joint_idx]) + tau_coupling[:, finger_idx, joint_idx]
                tau[:, self.fingers_mimic_joint[finger_idx][joint_idx]] = tau_el_mimic[:, finger_idx, joint_idx]
                joint_idx += 1
            finger_idx += 1

        self.oldTime = t
        return tau

    def init_constants(self):

        self.spring_k_each       = [0] * 5
        self.spring_k_mimic_each = [0] * 5
        self.fingers_syn_S       = [0] * 5
        for i, dim in enumerate(self.joints_dim):
            self.spring_k_each[i]       = [0] * dim  
            self.spring_k_mimic_each[i] = [0] * dim  
            self.fingers_syn_S[i]       = [0] * dim  

        # Init Spring K and Synergy coefficients
        self.spring_k_each[0][0] = 0.8  # k_thumb_j1
        self.spring_k_each[0][1] = 0.8  # k_thumb_j2
        self.spring_k_each[0][2] = 0.8  # k_thumb_j3
        self.spring_k_each[1][0] = 0.15 # k_index_j1
        self.spring_k_each[1][1] = 0.5  # k_index_j2
        self.spring_k_each[1][2] = 0.5  # k_index_j3
        self.spring_k_each[1][3] = 0.5  # k_index_j4
        self.spring_k_each[2][0] = 0.15 # k_middle_j1
        self.spring_k_each[2][1] = 0.5  # k_middle_j2
        self.spring_k_each[2][2] = 0.5  # k_middle_j3
        self.spring_k_each[2][3] = 0.5  # k_middle_j4
        self.spring_k_each[3][0] = 0.15 # k_ring_j1
        self.spring_k_each[3][1] = 0.5  # k_ring_j2
        self.spring_k_each[3][2] = 0.5  # k_ring_j3
        self.spring_k_each[3][3] = 0.5  # k_ring_j4
        self.spring_k_each[4][0] = 0.15 # k_little_j1
        self.spring_k_each[4][1] = 0.5  # k_little_j2
        self.spring_k_each[4][2] = 0.5  # k_little_j3
        self.spring_k_each[4][3] = 0.5  # k_little_j4

        # --- coefficents for S synergy matrix ---
        self.fingers_syn_S[0][0] = 2.5   # 3.0   # synS_thumb_j1
        self.fingers_syn_S[0][1] = 1.0   # 1.2   # synS_thumb_j2
        self.fingers_syn_S[0][2] = 1.25  # 1.2   # synS_thumb_j3
        self.fingers_syn_S[1][0] = -0.02 # -0.3  # synS_index_j1
        self.fingers_syn_S[1][1] = 1.5   # 1.57  # synS_index_j2
        self.fingers_syn_S[1][2] = 1.0   # 1.25  # synS_index_j3
        self.fingers_syn_S[1][3] = 1.75  # 0.8   # synS_index_j4
        self.fingers_syn_S[2][0] = 0.0           # synS_middle_j1
        self.fingers_syn_S[2][1] = 1.0   # 1.57  # synS_middle_j2
        self.fingers_syn_S[2][2] = 1.5   # 1.25  # synS_middle_j3
        self.fingers_syn_S[2][3] = 2.0   # 1.25  # synS_middle_j4
        self.fingers_syn_S[3][0] = 0.0           # synS_ring_j1
        self.fingers_syn_S[3][1] = 1.5   # 1.57  # synS_ring_j2
        self.fingers_syn_S[3][2] = 1.25  # 1.0   # synS_ring_j3
        self.fingers_syn_S[3][3] = 2.0   # 1.0   # synS_ring_j4
        self.fingers_syn_S[4][0] = 0.02          # synS_little_j1
        self.fingers_syn_S[4][1] = 1.5   # 1.57  # synS_little_j2
        self.fingers_syn_S[4][2] = 1.25          # synS_little_j3
        self.fingers_syn_S[4][3] = 2.0   # 1.0   # synS_little_j4

        # --- spring rates --- 
        self.spring_k_mimic_each[0][0] = 0.5 # k_thumb_j1_mimic
        self.spring_k_mimic_each[0][1] = 0.5 # k_thumb_j2_mimic
        self.spring_k_mimic_each[0][2] = 0.5 # k_thumb_j3_mimic
        self.spring_k_mimic_each[1][0] = 0.5 # k_index_j1_mimic
        self.spring_k_mimic_each[1][1] = 0.5 # k_index_j2_mimic
        self.spring_k_mimic_each[1][2] = 0.5 # k_index_j3_mimic
        self.spring_k_mimic_each[1][3] = 0.5 # k_index_j4_mimic
        self.spring_k_mimic_each[2][0] = 0.5 # k_middle_j1_mimic
        self.spring_k_mimic_each[2][1] = 0.5 # k_middle_j2_mimic
        self.spring_k_mimic_each[2][2] = 0.5 # k_middle_j3_mimic
        self.spring_k_mimic_each[2][3] = 0.5 # k_middle_j4_mimic
        self.spring_k_mimic_each[3][0] = 0.5 # k_ring_j1_mimic
        self.spring_k_mimic_each[3][1] = 0.5 # k_ring_j2_mimic
        self.spring_k_mimic_each[3][2] = 0.5 # k_ring_j3_mimic
        self.spring_k_mimic_each[3][3] = 0.5 # k_ring_j4_mimic
        self.spring_k_mimic_each[4][0] = 0.5 # k_little_j1_mimic
        self.spring_k_mimic_each[4][1] = 0.5 # k_little_j2_mimic
        self.spring_k_mimic_each[4][2] = 0.5 # k_little_j3_mimic
        self.spring_k_mimic_each[4][3] = 0.5 # k_little_j4_mimic

