import json
import logging
import random
import re
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

# -----------------------------
# Paths & constants
# -----------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_NAME = "gpt-4o-mini"

EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
ORDER_REGEX = re.compile(r'#?([A-Za-z]\d{3})')

SYSTEM_PROMPT = """
You are Sierra Outfitters' helpful outdoor gear assistant. 🌄
- Always be upbeat and reference the outdoors.
- You can: (1) check order status, (2) recommend products from our catalog,
  (3) offer an Early Risers 10% code between 8–10am PT on request.
- Ask clarifying questions when needed and keep responses concise.
"""


# -----------------------------
# Helper functions
# -----------------------------

def load_data() -> tuple[Dict[str, Any], list[dict]]:
    """Load orders and product catalog from JSON files.

    Returns:
        orders_by_key: dict keyed by "email,order_number" -> order dict
        product_catalog: list of product dicts
    """
    with open(DATA_DIR / "CustomerOrders.json") as f:
        customer_orders = json.load(f)

    # Build a lookup dict for fast (email, order_number) -> order
    orders_by_key: Dict[str, Any] = {}
    for order in customer_orders:
        email = order["Email"]
        order_number = order["OrderNumber"]
        key = f"{email},{order_number}"
        orders_by_key[key] = order

    with open(DATA_DIR / "ProductCatalog.json") as f:
        product_catalog = json.load(f)

    return orders_by_key, product_catalog


def extract_email(text: str) -> str | None:
    m = EMAIL_REGEX.search(text)
    return m.group(0) if m else None


def extract_order_number(text: str) -> str | None:
    m = ORDER_REGEX.search(text)
    if not m:
        return None

    core = m.group(1)
    core = core.upper()
    return f"#{core}" 


# -----------------------------
# SierraAgent class
# -----------------------------

class SierraAgent:
    def __init__(self, client: OpenAI, orders: Dict[str, Any], catalog: list[dict]):
        self.client = client
        self.orders = orders
        self.catalog = catalog
        self.conversation = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        self.discounts: set[str] = set()
        self.tracking_requested = False
        self.current_email: str | None = None
        self.current_order_number: str | None = None

    # ----- internal helpers -----

    def _respond(self, user_text: str | None = None, system_text: str | None = None) -> str:
        """Small helper to append messages and call the model."""
        if user_text is not None:
            self.conversation.append({"role": "user", "content": user_text})
        if system_text is not None:
            self.conversation.append({"role": "system", "content": system_text})

        resp = self.client.responses.create(
            model=MODEL_NAME,
            input=self.conversation,
        )
        reply = resp.output_text
        self.conversation.append({"role": "assistant", "content": reply})
        return reply

    def get_status(self, email: str, order_number: str) -> dict | None:
        key = f"{email},{order_number}"
        logger.debug("looking up order with key=%s", key)
        return self.orders.get(key)

    def classify_intent(self, user_input: str) -> str:
        logger.debug("classifying intent with LLM for input: %s", user_input)

        messages = [
            {
                "role": "system",
                "content": """
                You are an intent classifier for Sierra Outfitters customer chat.

                Given a single user message, decide whether their primary intent is:
                - TRACKING: they want order status, shipping status, or tracking info.
                - RECOMMENDATION: they want product suggestions, what to buy, gear for a trip, etc.
                - PROMOTION: they are explicitly asking for the Early Risers Promotion or a discount code.
                - OTHER: anything else (returns, general questions, small talk, etc.).

                Respond with exactly one word, in uppercase:
                TRACKING, RECOMMENDATION, PROMOTION, or OTHER.
                """,
            },
            {"role": "user", "content": user_input},
        ]

        resp = self.client.responses.create(
            model=MODEL_NAME,
            input=messages,
        )
        label = resp.output_text.strip().upper()
        logger.debug("raw intent label from LLM: %s", label)

        if "TRACK" in label:
            return "TRACKING"
        if "RECOMMEND" in label:
            return "RECOMMENDATION"
        if "PROMOTION" in label:
            return "PROMOTION"
        return "OTHER"


    # ----- tracking -----

    def get_tracking(self, user_input: str) -> str | None:
        email = extract_email(user_input)
        if email:
            self.current_email = email
            logger.debug("captured email: %s", email)

        order_number = extract_order_number(user_input)
        if order_number:
            self.current_order_number = order_number
            logger.debug("captured order number: %s", order_number)

        # If we have all info, do the lookup
        if self.tracking_requested and self.current_email and self.current_order_number:
            logger.debug("have both email and order number, performing lookup")
            order = self.get_status(self.current_email, self.current_order_number)
            logger.debug("order lookup result: %s", order)

            if order is not None:
                tracking_link = (
                    "https://tools.usps.com/go/TrackConfirmAction"
                    f"?tLabels={order['TrackingNumber']}"
                )

                order_context = f"""
                The customer previously requested tracking information and has now provided:
                - Customer email: {self.current_email}
                - Order number: {self.current_order_number}

                Here is the order data:
                - Status: {order['Status']}
                - Tracking number: {order['TrackingNumber']}
                - Tracking link: {tracking_link}

                Respond by clearly telling them their order status and giving them the tracking link,
                in a friendly, outdoorsy tone.
                """

                reply = self._respond(user_text=user_input, system_text=order_context)

                # Optionally reset the tracking state after success
                self.tracking_requested = False
                self.current_email = None
                self.current_order_number = None

                return reply
            else:
                # Order not found – let the model explain politely
                logger.debug("order not found for email=%s order_number=%s",
                             self.current_email, self.current_order_number)
                error_context = """
                You could not find an order matching the email and order number provided.
                Ask the customer to double-check for typos or provide a different email/order number.
                Keep the tone warm and outdoorsy.
                """
                return self._respond(user_text=user_input, system_text=error_context)

        # Missing email or order number – ask them for those details
        logger.debug("missing email or order number for tracking flow")
        error_context = """
        This is a customer service chat agent for Sierra Outfitters.

        If the customer's message doesn't clearly relate to tracking an order,
        getting product recommendations, or requesting the Early Risers Promotion,
        politely summarize what you can help with:

        - Check order status and provide tracking links (with email + order number).
        - Recommend products from the Sierra Outfitters catalog.
        - Offer a 10% Early Risers discount code between 8–10am PT.

        Then ask a short, friendly question to steer them toward one of these options.
        Keep the tone warm, outdoorsy, and concise.
        """
        return self._respond(user_text=user_input, system_text=error_context)

    # ----- recommendations -----

    def get_recommendations(self, user_input: str) -> str:
        # Answer customer questions based off of product catalogue
        logger.debug("generating recommendations based on catalog for input: %s", user_input)

        # Very simple keyword-based matching against product catalog
        tokens = [w.strip(".,!?").lower() for w in user_input.split()]
        matches: list[dict] = []
        for product in self.catalog:
            blob = json.dumps(product).lower()
            if any(tok in blob for tok in tokens):
                matches.append(product)

        if matches:
            top_products = matches[:5]
            source_description = "matching products from the catalog"
        else:
            top_products = self.catalog[:5]
            source_description = (
                "a sample of products from the catalog (no strong keyword matches were found)"
            )

        logger.debug("matched %d products for recommendations", len(matches))

        product_summaries = "\n".join(f"- {json.dumps(p)}" for p in top_products)

        system_context = f"""
        The customer is asking for product recommendations.

        You are an outdoorsy gear guide.

        You have the following {source_description} (in JSON form):
        {product_summaries}

        Follow these rules:
        1. If what the customer wants is still unclear, ask 1–2 short clarifying questions
        (trip type, weather, product type) and do NOT recommend specific products yet.
        2. If you have enough detail, recommend 1–3 products from the list above and briefly
        explain why, in an enthusiastic, outdoorsy tone.
        """

        return self._respond(user_text=user_input, system_text=system_context)

    def get_promotion(self, user_input: str) -> str:
        """
        Handle Early Risers Promotion requests.

        - If current time in PT is between 8:00 and 10:00 AM (inclusive of 8:00, exclusive of 10:00),
          generate a unique discount code and tell the user they get 10% off.
        - Otherwise, explain the valid hours and that they don't get the discount right now.
        """
        # Current time in Pacific Time
        now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
        logger.debug("current PT time for promotion: %s", now_pt.isoformat())

        start_hour = 8
        end_hour = 10  # treat as [8, 10)

        if start_hour <= now_pt.hour < end_hour:
            # Within promo window: generate a unique code
            def make_code() -> str:
                suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
                return f"EARLY-{suffix}"

            code = make_code()
            while code in self.discounts:
                code = make_code()
            self.discounts.add(code)
            logger.debug("generated promo code: %s", code)

            system_context = f"""
            The customer has requested an Early Risers Promotion and the current time
            in Pacific Time is {now_pt.strftime("%I:%M %p")}.

            You have generated the following unique 10% discount code for the customer:
            - {code}

            Explain that this code gives them 10% off as part of Sierra Outfitters' Early Risers promotion.
            Keep the tone outdoorsy, enthusiastic, and concise. Do NOT invent additional codes.
            Just clearly present this one code and how to use it.
            """
            return self._respond(user_text=user_input, system_text=system_context)
        else:
            # Outside promo window
            window_str = "8:00–10:00 AM Pacific Time"
            logger.debug(
                "promotion requested outside window: now_pt=%s (window=%s)",
                now_pt.isoformat(),
                window_str,
            )
            system_context = f"""
            The customer has requested an Early Risers Promotion, but the current time in Pacific Time
            is {now_pt.strftime("%I:%M %p")}, which is outside the promotion window of {window_str}.

            Politely explain that the Early Risers 10% discount is only available between {window_str},
            and that they are currently outside that window. Keep the tone warm, apologetic, and outdoorsy.
            Do NOT provide any discount code.
            """
            return self._respond(user_text=user_input, system_text=system_context)

    # ----- main chat -----

    def chat(self, user_input: str) -> str:
        logger.debug("user input: %s", user_input)

        # If we're already in a tracking flow (user asked earlier), keep collecting info
        if self.tracking_requested:
            logger.debug("continuing existing tracking flow")
            tracking_reply = self.get_tracking(user_input)
            if tracking_reply is not None:
                return tracking_reply

        # Otherwise, ask the LLM what the user wants
        intent = self.classify_intent(user_input)
        logger.debug("interpreted intent: %s", intent)

        if intent == "TRACKING":
            logger.debug("tracking has been requested; starting tracking flow")
            self.tracking_requested = True
            tracking_reply = self.get_tracking(user_input)
            if tracking_reply is not None:
                return tracking_reply

        if intent == "RECOMMENDATION":
            return self.get_recommendations(user_input)

        if intent == "PROMOTION":
            return self.get_promotion(user_input)

        # Default: generic chat with the main assistant
        system_context = f"""
        This is a customer service chat agent which can help with tracking ordered items 
        and getting recommendations on the product catalog. Kindly ask if the user
        needs help in either of these categories. Keep the tone warm and outdoorsy.
        """
        return self._respond(user_text=user_input, system_text=system_context)

# -----------------------------
# Entrypoint
# -----------------------------

def run_chat_loop():
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    client = OpenAI()

    orders, catalog = load_data()
    agent = SierraAgent(client, orders, catalog)

    print(
        "Sierra Outfitters Agent ready! "
        "What would you like help with today explorer? (Ctrl+C to exit)\n"
    )
    while True:
        try:
            user_input = input("Customer: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye, trailblazer! 🏔️")
            break

        if not user_input.strip():
            continue

        reply = agent.chat(user_input)
        print("Agent:", reply)


if __name__ == "__main__":
    run_chat_loop()
