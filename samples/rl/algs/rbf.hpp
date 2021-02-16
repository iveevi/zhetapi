#ifndef RBF_H_
#define RBF_H_

/*
 * Experience and replay buffer strategies.
 *
 * Includes:
 * 	- The experience structure
 * 	- Prioritized experience replay
 */

// Experience structure (s, a, s', r)
struct experience {
	int	index	= 0;
	bool	done	= false;
	double	reward	= 0;
	double	error	= 0;
	Vec	current	= {};
	Vec	next	= {};

	bool operator<(const experience &e) const {
		return error < e.error;
	}

	bool operator>(const experience &e) const {
		return error > e.error;
	}
};

// Priority replay buffer
class replays : public priority_queue <experience> {
	size_t	__size	= 0;
	size_t	__bsize	= 0;
public:
	replays(size_t size, size_t batch_size) : __size(size),
			__bsize(batch_size) {}

	vector <experience> sample() {
		vector <experience> b;

		for (size_t i = 0; i < __bsize; i++) {
			b.push_back(top());

			pop();
		}

		return b;
	}
	
	void add(const experience &e) {
		if (full())
			replace_bottom(e);
		else
			push(e);
	}

        void replace_bottom(const experience &e) {
                auto it_min = min_element(c.begin(), c.end());

                if (it_min->error < e.error) {
                        *it_min = e;

                        make_heap(c.begin(), c.end(), comp);
                }
        }

	bool full() {
		return (size() == __size);
	}
};

#endif
