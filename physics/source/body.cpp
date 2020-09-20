#include "body.h"

using namespace godot;

void Body::_register_methods() {
    register_method("_process", &Body::_process);

    register_property <Body, float> ("speed", &Body::speed, 0.01);
}

Body::Body() {
}

Body::~Body() {
    // add your cleanup here
}

void Body::_init() {
    // initialize any variables here
    time = 0.0;
}

void Body::_process(float delta) {
    time += delta;

    Godot::print("Body!");
    Vector3 new_position = Vector3(-0.01, -speed, 0);
    
    move_and_collide(new_position);
}
