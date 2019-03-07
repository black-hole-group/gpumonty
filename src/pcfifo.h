#ifndef _PCFIFO_H
#define _PCFIFO_H

#include <stdint.h>
#include <pthread.h>
#include <semaphore.h>

/*
 * pcfifo size is 2^16, which should be enough. This way, head and tail can be
 * declared as uint16_t, and we don't have to be update them with modulus
 * operator. (We use the overflow as modulus)
 */
#define PCFIFO_BUF_SIZE (1 << 16)

struct pcfifo
{
	void *buffer[PCFIFO_BUF_SIZE];
	uint16_t head, tail;
	sem_t empty;
	sem_t full;
	pthread_mutex_t lock;
};

/* Init an empty pcfifo */
void pcfifo_init(struct pcfifo *pcf);

/* Destroy the pcfifo
 * Note: elements stored at pcfifo should be freed */
void pcfifo_destroy(struct pcfifo *pcf);

/* Get the next element at pcfifo to be consumed*/
void* pcfifo_get(struct pcfifo *pcf);

/* Put a produced element in the pcfifo */
void pcfifo_put(struct pcfifo *pcf, void *element);

/* Put size produced elements in the pcfifo */
void pcfifo_put_many(struct pcfifo *pcf, void **element, unsigned size);

/*
 * Emit num_threads NULLs so that all consumers can end.
 * Note: consumers should check upon NULL elements and quit.
 */
void pcfifo_emit_end_tokens(struct pcfifo *pcf, unsigned num_threads);

#endif
