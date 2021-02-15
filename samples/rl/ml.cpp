#include "global.hpp"

// Networks

// On-line model
ml::DNN <double> model(6, {
	ml::Layer <double> (4, new ml::ReLU <double> ()),
	ml::Layer <double> (8, new ml::Sigmoid <double> ()),
	ml::Layer <double> (4, new ml::ReLU <double> ())
});

// Confidence model (percentage error estimator)
ml::DNN <double> confidence(6, {
	ml::Layer <double> (4, new ml::ReLU <double> ()),
	ml::Layer <double> (8, new ml::ReLU <double> ()),
	ml::Layer <double> (4, new ml::Sigmoid <double> ())
});
