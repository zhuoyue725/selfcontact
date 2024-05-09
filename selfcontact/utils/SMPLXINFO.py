# SMPLX_JOINT_NAMES_55 = [ # 1+54 身体全部部位
#     'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3', 'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist',
#     'jaw','left_eye_smplhf','right_eye_smplhf','left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3','right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'
# ]
LEFT_RIGHT_ARM_INDEX_55 = [
    13,14,16,17,18,19,20,21 # 16,17肩膀会与胸部交点
] + list(range(25, 55)) # 所有的手指

LEFT_RIGHT_LEG_INDEX_55 = [
    1,2,4,5,7,8,10,11
]

LEFT_RIGHT_ARM_INDEX_21 = [
    12,13,15,16,17,18,19,20
]

LEFT_RIGHT_LEG_INDEX_21 = [
    0,1,3,4,6,7,9,10
]

SMPLX_JOINT_NAMES_55 = [
    ('Hips', 0),
    ('LeftUpLeg', 1),
    ('RightUpLeg', 2),
    ('Spine', 3),
    ('LeftLeg', 4),
    ('RightLeg', 5),
    ('Spine1', 6),
    ('LeftFoot', 7),
    ('RightFoot', 8),
    ('Spine2', 9),
    ('LeftToe', 10),
    ('RightToe', 11),
    ('Neck', 12),
    ('LeftShoulder', 13),
    ('RightShoulder', 14),
    ('Head', 15),
    ('LeftArm', 16),
    ('RightArm', 17),
    ('LeftForeArm', 18),
    ('RightForeArm', 19),
    ('LeftHand', 20),
    ('RightHand', 21),
    ('jaw',22),
    ('left_eye_smplhf',23),
    ('right_eye_smplhf',24),
    ('LeftHandIndex1', 25),
    ('LeftHandIndex2', 26),
    ('LeftHandIndex3', 27),
    ('LeftHandMiddle1', 28),
    ('LeftHandMiddle2', 29),
    ('LeftHandMiddle3', 30),
    ('LeftHandPinky1', 31),
    ('LeftHandPinky2', 32),
    ('LeftHandPinky3', 33),
    ('LeftHandRing1', 34),
    ('LeftHandRing2', 35),
    ('LeftHandRing3', 36),
    ('LeftHandThumb1', 37),
    ('LeftHandThumb2', 38),
    ('LeftHandThumb3', 39),
    ('RightHandIndex1', 40),
    ('RightHandIndex2', 41),
    ('RightHandIndex3', 42),
    ('RightHandMiddle1', 43),
    ('RightHandMiddle2', 44),
    ('RightHandMiddle3', 45),
    ('RightHandPinky1', 46),
    ('RightHandPinky2', 47),
    ('RightHandPinky3', 48),
    ('RightHandRing1', 49),
    ('RightHandRing2', 50),
    ('RightHandRing3', 51),
    ('RightHandThumb1', 52),
    ('RightHandThumb2', 53),
    ('RightHandThumb3', 54),
]

SMPL_JOINT_NAMES = [ # 1+23
    'Pelvis', 'L_Hip',  'R_Hip','Spine1',  'L_Knee',  'R_Knee', 'Spine2',  'L_Ankle', 'R_Ankle',  'Spine3',  'L_Foot',  'R_Foot',  
    'Neck',  'L_Collar',  'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',  'L_Elbow',  'R_Elbow', 'L_Wrist',  'R_Wrist', 'L_Hand', 'R_Hand'
]

UPPER_BOFY_INDEX = [ # 上半身关节索引，总共21
    2,5,8,11,12,13,14,15,16,17,18,19,20
]

LOWER_BOFY_INDEX = [ # 下半身关节索引，总共21
    0,1,3,4,6,7,9,10
]

SMPLX_BODY_JOINT_NAMES = [ # SMPLX的21个身体关节θ(去除第一个pelvis)
    # "pelvis",
    "left_hip",
    "right_hip", # 1
    "spine1", # 2
    "left_knee",
    "right_knee", # 4
    "spine2",
    "left_ankle", # 6
    "right_ankle", # 7
    "spine3",
    "left_foot",
    "right_foot",
    "neck", # 11
    "left_collar",
    "right_collar", # 13
    "head",   # 14
    "left_shoulder",
    "right_shoulder", # 16
    "left_elbow",
    "right_elbow", # 18
    "left_wrist",
    "right_wrist", # 20
]

# 9上胸，6中胸，3肚子
# 13左肩，16左上臂，18 左小臂；14右肩，17右上臂，19 右小臂
# 12颈部，15头部，23左眼，24右眼
# 1左大腿，2右大腿

# 21+1个关节对应smplx的名称映射
JOINT_MAP_22 = {
    # 'BVH joint name': 'SMPLX joint index'
    'Hips': 0,
    'LeftUpLeg': 1,
    'RightUpLeg': 2,
    'Spine': 3,
    'LeftLeg': 4,
    'RightLeg': 5,
    'Spine1': 6,
    'LeftFoot': 7,
    'RightFoot': 8,
    'Spine2': 9,
    'LeftToe': 10,
    'RightToe': 11,
    'Neck': 12,
    'LeftShoulder': 13,
    'RightShoulder': 14,
    'Head': 15,
    'LeftArm': 16,
    'RightArm': 17,
    'LeftForeArm': 18,
    'RightForeArm': 19,
    'LeftHand': 20,
    'RightHand': 21,
    # 没有30个手指关节
    # 'LeftHandIndex1': 25,
    # 'LeftHandIndex2': 26,
    # 'LeftHandIndex3': 27,
    # 'LeftHandMiddle1': 28,
    # 'LeftHandMiddle2': 29,
    # 'LeftHandMiddle3': 30,
    # 'LeftHandPinky1': 31,
    # 'LeftHandPinky2': 32,
    # 'LeftHandPinky3': 33,
    # 'LeftHandRing1': 34,
    # 'LeftHandRing2': 35,
    # 'LeftHandRing3': 36,
    # 'LeftHandThumb1': 37,
    # 'LeftHandThumb2': 38,
    # 'LeftHandThumb3': 39,
    # 'RightHandIndex1': 40,
    # 'RightHandIndex2': 41,
    # 'RightHandIndex3': 42,
    # 'RightHandMiddle1': 43,
    # 'RightHandMiddle2': 44,
    # 'RightHandMiddle3': 45,
    # 'RightHandPinky1': 46,
    # 'RightHandPinky2': 47,
    # 'RightHandPinky3': 48,
    # 'RightHandRing1': 49,
    # 'RightHandRing2': 50,
    # 'RightHandRing3': 51,
    # 'RightHandThumb1': 52,
    # 'RightHandThumb2': 53,
    # 'RightHandThumb3': 54,
}