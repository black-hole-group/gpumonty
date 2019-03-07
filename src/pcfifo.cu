#include "pcfifo.h"

void pcfifo_init(struct pcfifo *pcf)
{
	pcf->head = 0;
	pcf->tail = 0;
	sem_init(&pcf->empty, 0, PCFIFO_BUF_SIZE);
	sem_init(&pcf->full, 0, 0);
	pthread_mutex_init(&pcf->lock, NULL);
}

void pcfifo_destroy(struct pcfifo *pcf)
{
	sem_destroy(&pcf->full);
	sem_destroy(&pcf->empty);
	pthread_mutex_destroy(&pcf->lock);
}

void *pcfifo_get(struct pcfifo *pcf)
{
	void *ret;

	sem_wait(&pcf->full);
	pthread_mutex_lock(&pcf->lock);
	ret = pcf->buffer[pcf->tail];
	pcf->tail++;
	pthread_mutex_unlock(&pcf->lock);
	sem_post(&pcf->empty);

	return ret;
}

void pcfifo_put(struct pcfifo *pcf, void *element)
{
	sem_wait(&pcf->empty);
	pthread_mutex_lock(&pcf->lock);
	pcf->buffer[pcf->head] = element;
	pcf->head++;
	pthread_mutex_unlock(&pcf->lock);
	sem_post(&pcf->full);
}

void pcfifo_put_many(struct pcfifo *pcf, void **element, unsigned size)
{
	for(unsigned i = 0; i < size; ++i) sem_wait(&pcf->empty);
	pthread_mutex_lock(&pcf->lock);
	for(unsigned i = 0; i < size; ++i) {
		pcf->buffer[pcf->head] = element[i];
		pcf->head++;
	}
	pthread_mutex_unlock(&pcf->lock);
	for(unsigned i = 0; i < size; ++i) sem_post(&pcf->full);
}


void pcfifo_emit_end_tokens(struct pcfifo *pcf, unsigned num_threads)
{
	for (int i = 0; i < num_threads; ++i) pcfifo_put(pcf, NULL);
}
