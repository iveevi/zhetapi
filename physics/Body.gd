extends KinematicBody

func _process(delta):
	var state = move_and_collide(-0.005 * Vector3.UP);
	
	if (state):
		print("!!!");
