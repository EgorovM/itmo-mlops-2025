import asyncio
import json
import logging
import os
from datetime import datetime

from flask import Flask, jsonify, request
from telegram import Bot
from telegram.error import TelegramError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set!")
    exit(1)

# Initialize Telegram bot
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Flask app
app = Flask(__name__)


def format_alert_message(alert_data):
    """Format alert message for Telegram"""
    try:
        alerts = alert_data.get("alerts", [])
        if not alerts:
            return "No alerts in the payload"

        messages = []
        for alert in alerts:
            status = alert.get("status", "unknown")
            labels = alert.get("labels", {})
            annotations = alert.get("annotations", {})

            # Determine icon based on severity and service
            if labels.get("service") == "ml_models":
                icon = "🤖" if status == "firing" else "✅"
            else:
                icon = "🔧" if status == "firing" else "✅"

            # Build message
            message_parts = [
                "{} **{}**".format(icon, status.upper()),
                "",
                "**Alert:** {}".format(annotations.get("summary", "No summary")),
            ]

            if labels.get("model_name"):
                message_parts.append("**Model:** {}".format(labels.get("model_name")))

            if labels.get("service"):
                message_parts.append("**Service:** {}".format(labels.get("service")))

            message_parts.extend(
                [
                    "**Severity:** {}".format(labels.get("severity", "unknown")),
                    "**Description:** {}".format(
                        annotations.get("description", "No description")
                    ),
                    "**Time:** {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "",
                    "---",
                ]
            )

            messages.append("\n".join(message_parts))

        return "\n".join(messages)

    except Exception as e:
        logger.error(f"Error formatting alert message: {e}")
        return f"Error formatting alert: {str(e)}"


async def send_telegram_message(message):
    """Send message to Telegram"""
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="Markdown"
        )
        logger.info("Message sent to Telegram successfully")
        return True
    except TelegramError as e:
        logger.error(f"Failed to send Telegram message: {e}")
        # Try without markdown formatting
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            logger.info("Message sent to Telegram without formatting")
            return True
        except TelegramError as e2:
            logger.error(f"Failed to send plain message: {e2}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error sending message: {e}")
        return False


@app.route("/webhook", methods=["POST"])
def webhook():
    """Webhook endpoint for Alertmanager"""
    try:
        # Get JSON data from Alertmanager
        alert_data = request.get_json()

        if not alert_data:
            logger.warning("Received empty alert data")
            return jsonify({"status": "error", "message": "No data received"}), 400

        logger.info(f"Received alert: {json.dumps(alert_data, indent=2)}")

        # Format and send message
        message = format_alert_message(alert_data)

        # Send message asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(send_telegram_message(message))
        loop.close()

        if success:
            return jsonify({"status": "success", "message": "Alert sent to Telegram"})
        else:
            return (
                jsonify({"status": "error", "message": "Failed to send to Telegram"}),
                500,
            )

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "bot_token_configured": bool(TELEGRAM_BOT_TOKEN),
            "chat_id_configured": bool(TELEGRAM_CHAT_ID),
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/test", methods=["POST"])
def test_message():
    """Test endpoint to send a test message"""
    try:
        test_message = "🧪 Test message from ML Monitoring Bot!\n\nIf you see this, the bot is working correctly."

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(send_telegram_message(test_message))
        loop.close()

        if success:
            return jsonify({"status": "success", "message": "Test message sent"})
        else:
            return (
                jsonify({"status": "error", "message": "Failed to send test message"}),
                500,
            )

    except Exception as e:
        logger.error(f"Error sending test message: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting Telegram Bot webhook server")
    logger.info("Bot token configured: {}".format(bool(TELEGRAM_BOT_TOKEN)))
    logger.info("Chat ID configured: {}".format(bool(TELEGRAM_CHAT_ID)))

    app.run(host="0.0.0.0", port=5000, debug=False)
