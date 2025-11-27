def compare_annotations(filename_predicted,filename_gt,verbose=0):
	p = open(filename_predicted,"rt")  	
	gt = open(filename_gt,"rt")  	
	all_lines_p = p.readlines()
	all_lines_gt = gt.readlines()

	#positions and tiles	
	number_lines_p = len(all_lines_p)
	number_lines_gt = len(all_lines_gt)

	match_positions = 1
	match_tiles = 1
	match_score = 1

	for i in range(number_lines_gt-1):		
		current_pos_gt, current_tile_gt = all_lines_gt[i].split()
		
		if verbose:
			print(i)
			print(current_pos_gt,current_tile_gt)

		try:
			current_pos_p, current_tile_p = all_lines_p[i].split()
			
			if verbose:
				print(current_pos_p,current_tile_p)

			if(current_pos_p != current_pos_gt):
				match_positions = 0
			if(current_tile_p != current_tile_gt):
				match_tiles = 0	
		except:
			match_positions = 0
			match_tiles = 0		
	try:
		#verify if there are more positions + tiles lines in the prediction file
		current_pos_p, current_tile_p = all_lines_p[i+1].split()
		match_positions = 0
		match_tiles = 0

		if verbose:
			print("EXTRA LINE:")
			print(current_pos_p,current_tile_p)
			
	except:
		pass



	points_positions = 0.04 * match_positions
	points_tiles = 0.03 * match_tiles	

	#scores
	last_line_p = all_lines_p[-1]
	score_p = last_line_p.split()
	last_line_gt= all_lines_gt[-1]
	score_gt = last_line_gt.split()
	
	if verbose:
		print(score_p,score_gt)

	if(score_p != score_gt):
		match_score = 0

	points_score = 0.01 * match_score

	return points_positions, points_tiles,points_score

def compare_annotations_bonus(filename_predicted,filename_gt,verbose=0):
	p = open(filename_predicted,"rt")  	
	gt = open(filename_gt,"rt")  	
	all_lines_p = p.readlines()
	all_lines_gt = gt.readlines()

	#positions and tiles	
	number_lines_p = len(all_lines_p)
	number_lines_gt = len(all_lines_gt)

	match_positions = 1
	match_tiles = 1
	match_score = 1


	for i in range(number_lines_gt-1):		
		current_pos1_gt, current_pos2_gt, current_tile_gt = all_lines_gt[i].split()
		
		if verbose:
			print(current_pos1_gt,current_pos2_gt,current_tile_gt)

		try:
			print(all_lines_p[i])
			current_pos_p1, current_pos_p2, current_tile_p = all_lines_p[i].split()
			if verbose:
				print(current_pos_p1, current_pos_p2, current_tile_p)

			if(current_pos_p1 != current_pos1_gt) or (current_pos_p2 != current_pos2_gt) :
				match_positions = 0
			if(current_tile_p != current_tile_gt):
				match_tiles = 0	
		except:
			match_positions = 0
			match_tiles = 0		
	try:
		#verify if there are more positions + tiles lines in the prediction file
		current_pos_p1, current_pos_p2, current_tile_p = all_lines_p[i+1].split()
		match_positions = 0
		match_tiles = 0

		if verbose:
			print("EXTRA LINE:")
			print(current_pos_p1, current_pos_p2, current_tile_p)
			
	except:
		pass



	points_positions = 0.02 * match_positions
	points_tiles = 0.02 * match_tiles	

	#scores
	last_line_p = all_lines_p[-1]
	score_p = last_line_p.split()
	last_line_gt= all_lines_gt[-1]
	score_gt = last_line_gt.split()
	
	if verbose:
		print(score_p,score_gt)

	if(score_p != score_gt):
		match_score = 0

	points_score = 0.01 * match_score

	return points_positions, points_tiles,points_score

#EVALUATION ON TEST TEST
print("EVALUATION ON TEST SET")

#change this on your machine pointing to your results (txt files)
predictions_path_root = "../fisiere_solutie/331_Alexe_Bogdan/"


#change this on your machine to point to the ground-truth test
gt_path_root = "../fake_test/ground-truth/"

#change this to 1 if you want to print results at each move
verbose = 0
total_points = 0
for game in range(1,2):
	#change this to range (1,6) on the test set
	for move in range(1,21):
		
		name_move = str(move)
		if(move< 10):
			name_move = '0'+str(move)

		filename_predicted = predictions_path_root + str(game) + '_' + name_move + '.txt'
		filename_gt = gt_path_root + str(game) + '_' + name_move + '.txt'

		game_move = str(game) + '_' + name_move
		points_position = 0
		points_tiles = 0
		points_score = 0		

		try:
			points_position, points_tiles, points_score = compare_annotations(filename_predicted,filename_gt,verbose)
		except:
			print("For image: ", game_move, " encountered an error")

		print("Image: ", game_move, "Points position: ", points_position, "Points tiles: ",points_tiles, "Points score: ", points_score)
		total_points = total_points + points_position + points_tiles + points_score

print(total_points)


#EVALUATION ON BONUS TEST TEST
print("EVALUATION ON BONUS TEST SET")


#change this on your machine pointing to your results (txt files)
predictions_path_bonus_root = "../fisiere_solutie/331_Alexe_Bogdan/bonus/"

#change this on your machine to point to the ground-truth test
gt_path_root_bonus = "../fake_test/ground-truth/bonus/"


#change this to 1 if you want to print results at each move
verbose = 1
total_points_bonus = 0
for game in range(1,2):
	for move in range(1,21):
		name_move = str(move)
		if(move< 10):
			name_move = '0'+str(move)

		filename_predicted = predictions_path_bonus_root + str(game) + '_' + name_move + '.txt'
		filename_gt = gt_path_root_bonus + str(game) + '_' + name_move + '.txt'

		game_move = str(game) + '_' + name_move
		points_position = 0
		points_tiles = 0
		points_score = 0		

		try:
			points_position, points_tiles, points_score = compare_annotations_bonus(filename_predicted,filename_gt,verbose)
		except:
			print("For image: ", game_move, " encountered an error")

		print("Image: ", game_move, "Points position: ", points_position, "Points tiles: ",points_tiles, "Points score: ", points_score)
		total_points_bonus = total_points_bonus + points_position + points_tiles + points_score

print(total_points)

print("Puncte totale din jocuri normale = ", total_points)
print("Puncte totale din bonus = ", total_points_bonus)
