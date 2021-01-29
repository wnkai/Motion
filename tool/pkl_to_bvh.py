import bpy
import random

# Globals
male_model_path = '/home/kaiwang/Documents/MpgModel/SMPL_unity_v.1.0.0/smpl/Models/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx'
female_model_path = '/home/kaiwang/Documents/MpgModel/SMPL_unity_v.1.0.0/smpl/Models/SMPL_f_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx'

bone_name_from_index = {
    0 : 'Pelvis',
    1 : 'L_Hip',
    2 : 'R_Hip',
    3 : 'Spine1',
    4 : 'L_Knee',
    5 : 'R_Knee',
    6 : 'Spine2',
    7 : 'L_Ankle',
    8: 'R_Ankle',
    9: 'Spine3',
    10: 'L_Foot',
    11: 'R_Foot',
    12: 'Neck',
    13: 'L_Collar',
    14: 'R_Collar',
    15: 'Head',
    16: 'L_Shoulder',
    17: 'R_Shoulder',
    18: 'L_Elbow',
    19: 'R_Elbow',
    20: 'L_Wrist',
    21: 'R_Wrist',
    22: 'L_Hand',
    23: 'R_Hand'
}

def main():
    # Setup scence
    scene = bpy.data.scenes['Scene']
    # Remove default cube
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    # Import gender specific .fbx template file
    bpy.ops.import_scene.fbx(filepath=male_model_path)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')

    armature = bpy.context.scene.objects['Armature']
    bones = armature.pose.bones

    for pose_bone in bones:
        print(pose_bone.location)
        x = pose_bone.location[0]
        y = pose_bone.location[1]
        z = pose_bone.location[2]
        pose_bone.location[0] = x
        pose_bone.location[1] = y
        pose_bone.location[2] = z

    bpy.ops.export_anim.bvh(filepath="output.bvh")

if __name__ == '__main__':
    for k, v in bone_name_from_index.items():
            bone_name_from_index[k] = 'm_avg_' + v
    main()