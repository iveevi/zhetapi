#ifndef BODY_H_
#define BODY_H_

#include <Godot.hpp>
#include <KinematicBody.hpp>
#include <KinematicCollision.hpp>

namespace godot {

class Body : public KinematicBody {
    GODOT_CLASS(Body, KinematicBody)

private:
    float time;
    float speed;

public:
    static void _register_methods();

    Body();
    ~Body();

    void _init(); // our initializer called by Godot

    void _process(float delta);
};

}

#endif
