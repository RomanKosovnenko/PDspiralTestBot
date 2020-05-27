import os
import urllib.parse
import urllib.request
import base64
import json

import requests

from botbuilder.core import ActivityHandler, MessageFactory, TurnContext, CardFactory
from botbuilder.schema import (
    ChannelAccount,
    HeroCard,
    CardAction,
    ActivityTypes,
    Attachment,
    AttachmentData,
    Activity,
    ActionTypes,
)

from predictor_helper import PredictorHelper


class PDbot(ActivityHandler):
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.

    async def on_members_added_activity(self, members_added: ChannelAccount, turn_context: TurnContext):
        for member_added in turn_context.activity.members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await self._send_welcome_message(turn_context)
        
    async def _send_welcome_message(self, turn_context: TurnContext):
        """
        Greet the user and give them instructions on how to interact with the bot.
        :param turn_context:
        :return:
        """
        reply = Activity(type=ActivityTypes.message)
        reply.text = "Hello and welcome! \n\n Send photo of the handwrited spiral to get probability of PD"
        reply.attachments = [await self._get_spiral_example(turn_context)]
        await turn_context.send_activity(reply)

    async def _get_spiral_example(self, turn_context: TurnContext) -> Attachment:
        """
        Creates an "Attachment" to be sent from the bot to the user at welcome message.
        :param turn_context:
        :return: Attachment
        """
        with open(
            os.path.join(os.getcwd(), "img/sparch.png"), "rb"
        ) as in_file:
            image_data = in_file.read()
        connector = await turn_context.adapter.create_connector_client(
            turn_context.activity.service_url
        )
        conversation_id = turn_context.activity.conversation.id
        response = await connector.conversations.upload_attachment(
            conversation_id,
            AttachmentData(
                original_base64=image_data,
                type="image/png",
            ),
        )
        base_uri: str = connector.config.base_url
        attachment_uri = (
            base_uri
            + ("" if base_uri.endswith("/") else "/")
            + f"v3/attachments/{response.id}/views/original"
        )
        return Attachment(
            content_type="image/png",
            content_url=attachment_uri,
        )

    async def on_message_activity(self, turn_context: TurnContext):
        """
        Handle all messages to the bot.
        Check is there any attachments in it. 
        If so, then handle it, otherwise send respons.
        :param turn_context:
        :return
        """
        if not await self._on_command_message_activity(turn_context):
            if (turn_context.activity.attachments and len(turn_context.activity.attachments) > 0):
                await self._handle_incoming_attachment(turn_context)
            else:
                await self._handle_no_attachments_message(turn_context)

    async def _on_command_message_activity(self, turn_context: TurnContext):
        if turn_context.activity.text == '/start':
            await self._send_welcome_message(turn_context)
            return True

    async def _handle_no_attachments_message(self, turn_context: TurnContext):
        """
        Response on messages without attachments
        :param turn_context:
        :return: string with promt to try one more time
        """
        reply = Activity(type=ActivityTypes.message)
        reply.text = "Your input was without attachment, please try again."

        await turn_context.send_activity(reply)

    async def _handle_incoming_attachment(self, turn_context: TurnContext):
        """
        Handle attachments uploaded by users. The bot receives an Attachment in an Activity.
        The activity has a List of attachments.
        Checks if the attachments is image. If so - handle image, otherwise - response with text message.
        :param turn_context:
        :return:
        """
        for attachment in turn_context.activity.attachments:
            if "image" in attachment.content_type:
                await self._handle_incoming_image_attachments_message(turn_context, attachment)
            else:
                await self._handle_no_image_attachments_message(turn_context)


            # attachment_info = await self._download_attachment_and_write(attachment)
            # if "filename" in attachment_info:
            #     await turn_context.send_activity(
            #         f"Attachment {attachment_info['filename']} has been received to {attachment_info['local_path']}"
            #     )

    async def _handle_no_image_attachments_message(self, turn_context: TurnContext):
        """
        Response on message with no image attachment
        :param turn_context:
        :return: string with promt to try one more time
        """
        reply = Activity(type=ActivityTypes.message)
        reply.text = "Your input was without image of spiral, please try again."

        await turn_context.send_activity(reply)

    async def _handle_incoming_image_attachments_message(self, turn_context: TurnContext, attachment: Attachment):
        """
        Retrieve the image attachment via the attachment's contentUrl and send it to model to analyze.
        :param attachment:
        :return
        """
        #TODO Add validation that input image is spiral
        reply = Activity(type=ActivityTypes.message)
        reply.text = "You send an image, please wait for results..."
        await turn_context.send_activity(reply)
        attachment_info = await self._download_attachment_and_write(attachment)
        await self._predict_pd(turn_context, attachment_info)
        

    async def _download_attachment_and_write(self, attachment: Attachment) -> dict:
        """
        Retrieve the attachment via the attachment's contentUrl.
        :param attachment:
        :return: Dict: keys "filename", "local_path"
        """
        try:
            response = urllib.request.urlopen(attachment.content_url)
            headers = response.info()

            # If user uploads JSON file, this prevents it from being written as
            # "{"type":"Buffer","data":[123,13,10,32,32,34,108..."
            if headers["content-type"] == "application/json":
                data = bytes(json.load(response)["data"])
            else:
                data = response.read()

            local_filename = os.path.join(os.getcwd(), attachment.name)
            with open(local_filename, "wb") as out_file:
                out_file.write(data)

            return {"filename": attachment.name, "local_path": local_filename}
        except Exception as exception:
            print(exception)
            return {}

    async def _predict_pd(self, turn_context: TurnContext, attachment_info: dict): 
        predictorHelper = PredictorHelper()

        predictions = predictorHelper.get_prediction(attachment_info)

        for prediction in predictions:
            reply = Activity(type=ActivityTypes.message)
            reply_str = ""
            if(len(prediction["models"]) == 0):
                reply_str += f"\n\nThe {prediction['predictor']} has no models yet \n\n"
            else: 
                reply_str += f"\n\n The {prediction['predictor']} predictor use model(s) and get result(s):\n\n"
            for model in prediction["models"]:
                reply_str += f"\t{model['model']} model says result is: {model['prediction']}\n\n"
            reply.text = reply_str
            await turn_context.send_activity(reply)
        


        

        
    

    