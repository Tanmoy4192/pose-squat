# #Squat Analyzer with Reference Pose Matching
# 1. Initialize variables

# rep_count = 0
# stage = "UP"

# 2. For every frame do
# user_landmarks = get_pose_from_camera()
# reference_landmarks = get_pose_from_reference_video()

# 3. Get important joints from user pose

# left_shoulder 
# right_shoulder 

# left_hip 
# left_knee 
# left_ankle 
# right_ankle 

# 4. Get joints from reference pose
# ref_hip 
# ref_knee
# ref_ankle 

# 5. Calculate knee angle for user
# knee_angle = angle(left_hip , left_knee , left_ankle)

# 6. Calculate knee angle for reference
# ref_knee_angle = angle(ref_hip , ref_knee , ref_ankle)

# 7. Check leg distance
# shoulder_width = distance(left_shoulder , right_shoulder)
# feet_width = distance(left_ankle , right_ankle)
# ratio = feet_width / shoulder_width

# if ratio < 0.8 OR ratio > 1.2
#     pose_correct = False
#     feedback = "Keep legs shoulder width"

# 8. Detect squat down position
# if knee_angle < 100
#     stage = "DOWN"

# 9. Detect squat up position
# if knee_angle > 160 AND stage == "DOWN"
#     rep_count = rep_count + 1
#     stage = "UP"

# 10. Compare user pose with reference pose

# angle_difference = abs(knee_angle - ref_knee_angle)
# similarity_score = 1 / (1 + angle_difference)

# 11. Check similarity
# if similarity_score < 0.8
#     pose_correct = False
#     feedback = "Follow the reference squat"
#     play_reference_video = False
# else
#     play_reference_video = True

# 12. Decide skeleton color

# if pose_correct == True
#     skeleton_color = WHITE
# else
#     skeleton_color = RED

# 13. Return results
# return rep_count, knee_angle
