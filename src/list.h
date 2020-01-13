#ifndef LIST_H
#define LIST_H

// Custom Built Libraries
#include <cstdlib>

namespace trees {
	// Beginning of list class
	template <typename data_t>
	struct list {
		data_t *curr;
		list *next;

                std::size_t size();
		std::size_t get_index(data_t *);

		list *operator()(data_t *);

                list *get(std::size_t);
		list *operator[](std::size_t);
	};

        template <typename data_t>
        std::size_t list <data_t> ::size()
        {
                auto *cpy = this;
                std::size_t counter = 0;

                while (cpy != nullptr) {
                        //std::cout << "#" << counter + 1 << " @";
                        //std::cout << cpy << std::endl;
                        //std::cout << "[" << counter << "] @";
                        //std::cout << this->get(counter) << " - [";
			//std::cout << cpy << "]" << " which contains @";
			//std::cout << cpy->curr << std::endl;
			cpy = cpy->next;
                        counter++;
                }

                return counter;
        }
	
	template <typename data_t>
	std::size_t list <data_t> ::get_index(data_t *nd)
        {
		auto *cpy = this;
		int index = 0;

		while (cpy != nullptr) {
			if (*(cpy->curr->dptr) == *(nd->dptr))
				return index;
			cpy = cpy->next;
			index ++;
		}

		return - 1;
	}

	template <typename data_t>
	list <data_t> *list <data_t> ::operator()(data_t *nd)
	{
		list <data_t> *cpy = this;

		while (cpy != nullptr) {
			if (*(cpy->curr->tptr) == *(nd->tptr))
				return cpy;
			cpy = cpy->next;
		}

		return nullptr;
	}

        template <typename data_t>
        list <data_t> *list <data_t> ::get(std::size_t i)
	{
		auto *cpy = this;
		int index = i;

		while (index > 0) {
			if (cpy == nullptr)
				return nullptr;
			cpy = cpy->next;
			index --;
		}

		return cpy;
	}

	template <typename data_t>
	list <data_t> *list <data_t> ::operator[](std::size_t i)
	{
		list <data_t> *cpy = this;
		int index = i;

		while (index >= 0) {
			if (cpy == nullptr)
				return nullptr;
			cpy = cpy->next;
			index --;
		}

		return cpy;
	}
}

#endif