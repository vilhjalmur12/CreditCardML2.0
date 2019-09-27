import pika
from pika.exceptions import StreamLostError
import json
import os


from Services.Log.Log import Log

logger = Log(file_handler=True, file='./logs/logs.log')


class Controller:
    def __init__(self):
        self.create_connection()

        self.training_queue = "ml.trainingQueue"
        self.reporting_queue = 'ml.reportingQueue'

        self.channel.queue_declare(queue=self.training_queue, durable=True)
        self.channel.queue_declare(queue=self.reporting_queue, durable=True)


        self.channel.basic_consume(queue=self.training_queue, on_message_callback=self.TrainingCallback, auto_ack=True)

    def deployment_callback(self, ch, method, properties, body):
        deployment_info = json.loads(body)


    def TrainingCallback(self, ch, method, properties, body):
        info_obj = json.loads(body)


    def ReportingCallback(self, ch, method, properties, body):
        info_obj = json.loads(body)
        logger.info(__name__, 'Recieved deployment job for {}'.format(info_obj['client']))
        logger.debug(__name__, '{}'.format(info_obj))
        self._deployService.send_deployment(info_obj['service'], info_obj)


    def create_connection(self):
        RABBIT = os.environ['RABBIT']
        RABBIT_USER = os.environ['RABBIT_USER']
        RABBIT_PASS = os.environ['RABBIT_PASS']
        RABBIT_VHOST = os.environ['RABBIT_MQ_VHOST']

        self._deployService = DeploymentService()

        # RabbitMQ
        credentials = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(RABBIT, 5672, RABBIT_VHOST, credentials))
        self.channel = self.connection.channel()

if __name__ == '__main__':
    logger.info(__name__, 'Deployment Service Started')
    running = True
    reconnect_tries = 0
    while running and reconnect_tries <= 5:
        try:
            controller = Controller()
            controller.channel.start_consuming()
        except StreamLostError:
            logger.error(__name__, 'Error with AMQP stream connection')
            reconnect_tries += 1
            controller.connection.close()
        except KeyboardInterrupt:
            logger.info(__name__, 'Manually closed training scheduler, goodbye!')
            running = False

    controller.connection.close()