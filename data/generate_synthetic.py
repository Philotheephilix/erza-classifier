import numpy as np
import pandas as pd
from typing import Optional
from data.feature_schema import FEATURE_COLUMNS, CATEGORY_LABELS, APP_CATEGORY_MAP
from data.sender_resolution import MESSAGING_PACKAGES, EMAIL_PACKAGES

# ─────────────────────────── App / package pool ────────────────────────────

_APPS = [
    # (package_name, display_name, app_category, can_be_personal_sender)
    ("com.whatsapp", "WhatsApp", "messaging", True),
    ("org.telegram.messenger", "Telegram", "messaging", True),
    ("com.discord", "Discord", "messaging", True),
    ("com.google.android.apps.messaging", "Messages", "messaging", True),
    ("org.thoughtcrime.securesms", "Signal", "messaging", True),
    ("com.instagram.android", "Instagram", "social", True),
    ("com.twitter.android", "Twitter", "social", False),
    ("com.linkedin.android", "LinkedIn", "social", False),
    ("com.reddit.frontpage", "Reddit", "social", False),
    ("com.snapchat.android", "Snapchat", "messaging", True),
    ("in.amazon.mShop.android.shopping", "Amazon", "shopping", False),
    ("app.zomato", "Zomato", "shopping", False),
    ("in.swiggy.android", "Swiggy", "shopping", False),
    ("com.flipkart.android", "Flipkart", "shopping", False),
    ("com.phonepe.app", "PhonePe", "finance", False),
    ("net.one97.paytm", "Paytm", "finance", False),
    ("com.google.android.gm", "Gmail", "productivity", True),
    ("com.slack", "Slack", "productivity", True),
    ("com.microsoft.teams", "Teams", "productivity", True),
    ("com.google.android.calendar", "Calendar", "productivity", False),
    ("com.google.android.youtube", "YouTube", "media", False),
    ("tv.twitch.android.app", "Twitch", "media", False),
    ("com.android.systemui", "System", "system", False),
]

# ─────────────────── Text templates per category ───────────────────────────

_TEXTS = {
    "message": [
        # Direct personal messages
        ("Mom", "Are you coming home this weekend?"),
        ("Dad", "Call me when you're free, it's important"),
        ("Priya", "Hey! What are you up to tonight?"),
        ("Rahul", "The assignment is due tomorrow, did you finish it?"),
        ("Sarah", "Can you pick me up at 6pm?"),
        ("Alex", "yo check this out lol"),
        ("Ananya", "Happy birthday!! 🎂 hope you have a great day"),
        ("Kiran", "Are you okay? You seemed off today"),
        ("Mike", "Meeting got pushed to 3pm"),
        ("Jess", "Miss you! When are you visiting?"),
        ("Nidhi", "I'm running 10 mins late, start without me"),
        ("Arjun", "Can you lend me your notes from yesterday?"),
        ("Divya", "Just saw your post, it made my day ❤️"),
        ("Sam", "Did you catch the match last night??"),
        ("Pooja", "Can we reschedule our call to 5pm?"),
        # Greetings and pleasantries
        ("Mom", "Good morning beta! Have a great day ☀️"),
        ("Partner", "Good night love 💕 sleep well"),
        ("Best Friend", "Morning!! ready for today??"),
        ("Colleague", "Hey, hope you're doing well! Quick question…"),
        ("Friend", "Hii! How are you? Haven't talked in so long"),
        ("Cousin", "Happy Diwali! 🪔 Lots of love"),
        ("Friend", "Happy New Year!! 🎉🎊"),
        ("Sister", "Happy birthday to you!! 🥳🎂 Have the best day!"),
        ("Mom", "Did you eat? Take care of yourself"),
        ("Dad", "Drive safe, let me know when you reach"),
        # Casual conversation
        ("Friend", "omg you have to see this 😂"),
        ("Roommate", "Can you grab milk on your way back?"),
        ("Partner", "What do you want for dinner?"),
        ("Friend", "Did you see what happened?? 😱"),
        ("Classmate", "bro what was the answer to Q3?"),
        ("Friend", "you up?"),
        ("Colleague", "Hey are you free for a quick call?"),
        ("Friend", "This song reminds me of you lol"),
        ("Sibling", "can you please not use all the hot water"),
        ("Friend", "we should definitely do this again sometime"),
        # T5 spam / cold outreach / unsolicited (open_rate ~0.02-0.05 in training)
        ("Insurance Agent", "Hello sir/ma'am, I'm calling about a special insurance plan for you"),
        ("Unknown", "Congratulations! You have been selected for a prize. Click to claim"),
        ("Marketing", "Hi, we offer loans up to ₹10 lakhs. Interested?"),
        ("Recruiter", "Exciting opportunity! We have a position that matches your profile"),
        ("Sales", "Limited offer: invest in our scheme and get 30% returns"),
        ("Unknown Number", "Please call back on this number urgently. Very important."),
        ("Job Portal", "10 new jobs match your profile. Apply now"),
        ("Agent", "Property investment opportunity in prime location. Call now"),
        ("Spam", "You've won a free gift! Reply YES to claim"),
        ("MLM Company", "Join our team and earn from home. Full training provided"),
        # Work/office messages (real names, not titles)
        ("Shreya", "Quick question about the report"),
        ("Aditya", "Hey can you review my PR when you get a chance?"),
        ("Rohan", "Sprint planning at 2pm — can you make it?"),
        ("Vikram", "Can we talk about the project timeline?"),
        ("Kavitha", "Reminder: timesheet submission due today"),
        ("Sanjay", "The meeting has been moved to Room 3"),
        ("Pradeep", "Great work on the presentation yesterday!"),
        ("Neha", "Hi, I had a quick question about the codebase"),
        # Coordination / planning asks — real-time reply needed
        ("Shiva", "When is the meet"),
        ("Rahul", "Are you coming today?"),
        ("Priya", "When are you reaching? We're all here"),
        ("Ankit", "Still on for tonight?"),
        ("Neha", "Shall we meet at 7? Confirm?"),
        ("Vikram", "You joining us or not? Let us know"),
        ("Kapil", "You in for Saturday?"),
        ("Arjun", "Are you coming to the meetup on Friday?"),
        ("Sneha", "When is the plan? Are you free Sunday?"),
        ("Partner", "Are you free this evening? Let's plan"),
        ("Mom", "When are you coming home?"),
        ("Best Friend", "What's the plan for tonight?"),
        ("Rohan", "Let's meet up this week — when are you free?"),
        ("Amit", "Catch up soon? It's been a while"),
        ("Kiran", "Link up tomorrow? You around?"),
        ("Meera", "Can you come to the meetup?"),
        ("Raj", "You coming or not? tell me now"),
        ("Sanya", "Confirmed for tomorrow?"),
        # DMs via social apps (Instagram, Twitter, Discord, Snapchat)
        ("rahul_sharma23", "Hey! Are you coming to the meetup?"),
        ("priya_travels", "Are you free this weekend? Let's plan something"),
        ("sneha_foodie", "Bhai check this place out — you'll love it"),
        ("karan_memes", "This is literally you 😂 had to send it"),
        ("Neha Sharma", "Neha Sharma sent you a Direct Message."),
        ("anjali_design", "Love your latest post btw! 😍"),
        ("Discord DM", "yo are you around? need to talk"),
        ("rahul_dev", "Can you hop on voice real quick?"),
        ("u/techguru", "u/techguru sent you a message: loved your comment"),
        # Group chats
        ("Group: Family", "Dad bought the tickets! We're going Saturday"),
        ("Work Group", "Anyone free for lunch?"),
        ("College Group", "Exam results are out!"),
        ("Group: Friends", "Game night this Saturday, who's in?"),
        ("Dev Group", "Server is down, can someone check?"),
        ("Group: Family", "Family reunion next month, save the date!"),
        ("Office Group", "Potluck this Friday, sign up for dishes"),
        ("Study Group", "Notes for chapter 5 uploaded in drive"),
        ("Friends Group", "who's coming to the concert??"),
        ("Flatmates", "someone left dishes in the sink again 🙄"),
    ],
    "email": [
        # Work emails (real person names used in Gmail/Outlook)
        ("Priya Sharma", "Regarding your application — next steps"),
        ("Sneha Iyer", "Your offer letter is ready"),
        ("Vikram", "Quick question about the Q3 report"),
        ("Rohan Mehta", "Following up on yesterday's meeting"),
        ("Kavitha", "Performance review scheduled for Friday"),
        ("Aditya Kumar", "Sprint review notes attached"),
        ("Shruti", "Can you review this document before EOD?"),
        ("Nikhil", "All-hands meeting rescheduled to Thursday"),
        # System/service emails
        ("Prof. Kumar", "Important: exam schedule change"),
        ("Support Team", "Your ticket #4532 has been resolved"),
        ("IT Support", "Your password expires in 3 days"),
        ("Security Alert", "Unusual sign-in detected on your account"),
        ("No-Reply", "Please verify your email address"),
        # Finance/transactional emails
        ("Accounts Team", "Invoice #2031 payment confirmation"),
        ("HDFC Bank", "Your monthly statement is ready"),
        # Newsletters / marketing
        ("Tech Weekly", "This week in tech — top stories"),
        ("Shruti at Startup", "Exciting opportunity at a fast-growing startup"),
        ("Medium", "Top stories for you today"),
        ("Product Hunt", "Today's top products: AI tools for dev"),
        # Personal emails (Gmail — just the person's name shows as title)
        ("Mom", "Hi beta, how are you doing? Video call this Sunday?"),
        ("Rahul", "Hey! Sending you the photos from last trip"),
        ("IIT Alumni Network", "Save the date: Reunion 2026"),
    ],
    "social": [
        # Instagram — feed/social (low urgency, low importance)
        ("Instagram", "priya_travels liked your photo."),
        ("Instagram", "rahul_sharma23 and 47 others liked your post."),
        ("Instagram", "Alex started following you"),
        ("Instagram", "New story from someone you follow"),
        ("Instagram", "Your reel reached 1,000 views!"),
        ("Instagram", "Your reel has been shared 100 times! 🎉"),
        ("Instagram", "Rahul mentioned you in a comment"),
        ("Instagram", "Your photo got 50 likes in the last hour"),
        ("Instagram", "5 new followers this week"),
        ("Instagram", "You're on a streak with priya_travels! Keep posting 🔥"),
        ("Instagram", "Techburner is live now! Join the stream."),
        ("Instagram", "CarryMinati posted for the first time in a while."),
        ("Instagram", "Your story was seen by 312 people."),
        ("Instagram", "Your reel reached 10,000 views! 🎉"),
        ("Instagram", "You haven't posted in 14 days. Your followers miss you!"),
        # Snapchat — social feed
        ("Snapchat", "You have a new snap from Ananya"),
        ("Snapchat", "kavya_snap opened your Snap! 👀"),
        ("Snapchat", "Neha took a screenshot of your Snap!"),
        ("Snapchat", "Your streak with Sam is about to end! 🔥"),
        ("Snapchat", "Priya posted to their story"),
        ("Snapchat", "Your memories from 1 year ago"),
        ("Snapchat", "karan_memes added you as a friend!"),
        ("Snapchat", "You have a new friend suggestion: Sneha Patel"),
        # Twitter/X
        ("Twitter", "techguru_india retweeted your post."),
        ("Twitter", "Priya Sharma liked your tweet."),
        ("Twitter", "You have 5 new mentions"),
        ("Twitter", "Rahul retweeted your post"),
        ("Twitter", "Trending near you: #IPL2026"),
        ("Twitter", "Trending in India: #Budget2026 — 1.4M tweets"),
        ("Twitter", "Your tweet got 100 likes"),
        ("Twitter", "Your tweet got 500 likes in the last hour! 🔥"),
        ("Twitter", "Thread alert: Kunal Shah posted a long thread on personal finance."),
        ("X (Twitter)", "Community note added to a post you liked."),
        ("Twitter", "karan_memes started following you."),
        # LinkedIn — social feed (low urgency)
        ("LinkedIn", "Your post got 48 new impressions"),
        ("LinkedIn", "3 people viewed your profile today"),
        ("LinkedIn", "Congratulate Priya on her new job!"),
        ("LinkedIn", "You appeared in 12 searches this week"),
        ("LinkedIn", "New connection request from a mutual"),
        ("LinkedIn", "Your application was viewed by the recruiter"),
        ("LinkedIn", "Your post is getting attention! 48 reactions so far."),
        ("LinkedIn", "Rahul Kumar sent you a connection request."),
        ("LinkedIn", "Amit Verma commented: Really insightful, thanks for sharing!"),
        ("LinkedIn", "New job alert: Senior Software Engineer at Zepto — Bangalore"),
        ("LinkedIn", "Today is Sneha Patel's work anniversary! 🎊"),
        ("LinkedIn", "3 people from Google viewed your profile today."),
        ("LinkedIn", "Ananya endorsed you for: Machine Learning"),
        ("LinkedIn", "Congratulate Rahul on his new job at Swiggy!"),
        ("LinkedIn", "Your profile strength is All-Star!"),
        ("LinkedIn", "47 people viewed your profile this week — a new record!"),
        # Google Photos
        ("Google Photos", "New memories created from your photos"),
        ("Google Photos", "Priya shared an album with you"),
        ("Google Photos", "Your photos from Goa trip are ready"),
        ("Google Photos", "Monthly photo recap is ready"),
        # Reddit
        ("Reddit", "Your comment got 23 upvotes"),
        ("Reddit", "Your comment got 47 upvotes."),
        ("Reddit", "Your post in r/india is trending! 🔥"),
        ("Reddit", "Hot in r/technology: new AI breakthrough"),
        ("Reddit", "Hot in r/learnprogramming: I went from 0 to SDE2 in 18 months"),
        ("Reddit", "Someone replied to your post"),
        ("Reddit", "New posts in r/india you might like"),
        ("Reddit", "New post in r/bangalore you might like."),
        ("Reddit", "Weekly digest: Top posts from your communities."),
        ("Reddit", "You received a Reddit Gold award on your post! 🏅"),
        ("Reddit", "u/mod_india removed your post."),
        # YouTube
        ("YouTube", "Techburner just uploaded: iPhone 18 Pro — First Impressions!"),
        ("YouTube", "MKBHD uploaded: The Best Phones of 2026"),
        ("YouTube", "Congrats! Your channel reached 1,000 subscribers 🎉"),
        ("YouTube", "Your video was liked 500 times today!"),
        ("YouTube", "Recommended for you: 10 productivity hacks that changed my life"),
        ("YouTube", "A channel you love just posted after 6 months."),
        ("YouTube", "Your comment got a heart from the creator ❤️"),
        ("YouTube", "Live premiere starting in 15 minutes: Dhruv Rathee"),
        ("YouTube", "Congrats! Your video reached 10,000 views 🎊"),
        # Facebook
        ("Facebook", "Rohan Mehta commented: Looking amazing! 😍"),
        ("Facebook", "Priya Sharma and 12 others reacted to your photo."),
        ("Facebook", "Amit Kumar tagged you in a photo."),
        ("Facebook", "Today's birthday: Priya"),
        ("Facebook", "Vikram Singh invited you to: Diwali House Party 2026 🪔"),
        ("Facebook", "Memories: See what you posted 3 years ago."),
        ("Facebook", "Your video got 1,000 views!"),
        ("Facebook", "Priya Sharma poked you!"),
        ("Facebook", "Rahul Mehta commented on a post you're tagged in."),
        # Security alerts (high urgency/importance)
        ("Instagram", "A login from an unrecognized device was detected."),
        ("Instagram", "Your account will be restricted unless you verify your identity."),
        ("Facebook", "We noticed an unusual login attempt on your account."),
        ("Facebook", "Security alert: someone tried to access your account."),
        ("Twitter", "Your account has a new login from a new device."),
        ("Twitter", "Your account may have been compromised. Change your password now."),
        ("LinkedIn", "Security alert: your account was accessed from a new device."),
        ("Snapchat", "Your account was logged in from an unrecognized device."),
        # Pinterest
        ("Pinterest", "priya_creates saved your Pin."),
        ("Pinterest", "12 people saved your board this week!"),
        ("Pinterest", "New follower: sneha_diy"),
        ("Pinterest", "Trending in Home Decor: Japandi style ideas"),
        # ShareChat / Moj / Josh (Indian short video)
        ("ShareChat", "priya_sharma liked your post!"),
        ("ShareChat", "Your post got 1,000 views! 🎉"),
        ("ShareChat", "rahul_memes commented: Too funny bhai 😂"),
        ("Moj", "sneha_dance started following you on Moj!"),
        ("Moj", "Your video is trending in your area! 🔥"),
        ("Josh", "priya_josh liked your video."),
        ("Josh", "Your Josh video reached 5,000 views!"),
        # Discord — server/group notifications
        ("Discord", "New message in Dev Squad · #backend"),
        ("Discord", "Among Us Night — starting in 10 mins! Join #gaming"),
        ("Discord", "karan_memes sent you a friend request."),
        ("Discord", "You were mentioned in #announcements: @everyone server maintenance tonight"),
        ("Discord", "Your friend rahul_dev is now online."),
        ("Discord", "You have 5 unread messages in 3 servers."),
        # Facebook
        ("Facebook", "Rahul and 3 others commented on your post"),
        ("Facebook", "You have 2 new friend requests"),
        ("Facebook", "Today's birthday: Priya"),
        ("Facebook", "Ananya tagged you in a photo"),
    ],
    "promo": [
        # Food delivery promos
        ("Zomato", "50% off your next order! Use SAVE50"),
        ("Zomato", "Craving something? Order now and save ₹100"),
        ("Zomato", "Free delivery on orders over ₹199 tonight"),
        ("Swiggy", "Free delivery on orders above ₹299"),
        ("Swiggy", "It's raining discounts! Order in the next hour"),
        ("Swiggy", "Flat 30% off on your favourite restaurants"),
        # E-commerce promos
        ("Amazon", "Flash sale: Up to 70% off electronics today only"),
        ("Amazon", "Your wishlist item is now on sale"),
        ("Amazon", "Last chance: deals end in 2 hours"),
        ("Flipkart", "Big Billion Days is back! Don't miss out"),
        ("Flipkart", "Flat 40% off on mobiles today only"),
        ("Myntra", "End of reason sale: flat 60% off on top brands"),
        ("Myntra", "New arrivals in your style preferences"),
        ("AJIO", "New arrivals: summer collection is here"),
        # Travel and services
        ("MakeMyTrip", "Flight prices drop alert for your saved route"),
        ("Uber", "Get ₹50 off your next ride with RIDE50"),
        ("Ola", "Weekend special: 20% cashback on cab rides"),
        # General promos
        ("Nykaa", "Buy 2 get 1 free on skincare — today only"),
        ("Paytm", "Scratch card: you may have won ₹500!"),
        ("Dunzo", "Your neighborhood store now on Dunzo"),
        ("BookMyShow", "Early bird offers for upcoming movies"),
        ("Hotstar", "New show alert: The Grand Tour is now streaming"),
        ("BigBasket", "Fresh morning deals: 20% off on vegetables"),
        ("PharmEasy", "Medicine reminder + 15% off on reorder"),
        # Time-urgency-style promos that still should NOT be urgent
        ("Swiggy", "Free delivery for the next 2 hours — order now"),
        ("MakeMyTrip", "Special offer on flights this weekend only"),
        ("Zomato", "Hurry! Limited time: 40% off expires soon"),
        ("Amazon", "Lightning deal: ends in 30 minutes"),
        ("Myntra", "Sale ends tonight — last chance to grab deals"),
        ("OYO", "Weekend rates drop 40% — book by midnight"),
        ("Hotstar", "Watch free for next 7 days — sign up now"),
        ("Paytm", "Offer ending: cashback on next 3 transactions"),
        ("Ola", "Only 5 minutes left to redeem your promo code"),
        ("Zepto", "Flash sale: groceries in 10 mins at 30% off"),
    ],
    "transaction": [
        # Regular successful transactions
        ("PhonePe", "You paid ₹500 to Priya via UPI"),
        ("Paytm", "₹1,200 debited from your account"),
        ("HDFC Bank", "Your salary credit ₹45,000 received"),
        ("PhonePe", "Rahul sent you ₹250"),
        ("Amazon Pay", "Refund of ₹899 processed successfully"),
        ("PhonePe", "UPI payment of ₹150 to Zomato successful"),
        ("Paytm", "Bill payment of ₹1,200 for electricity done"),
        ("Jupiter", "₹300 cashback credited to your account"),
        ("HDFC Bank", "Credit card bill due: ₹8,450 by Nov 5"),
        ("GPay", "You have a pending payment request from Ananya"),
        ("PhonePe", "UPI transaction successful: ₹800 to Swiggy"),
        ("SBI Bank", "₹2,500 withdrawn from ATM"),
        ("ICICI Bank", "Your credit card was used for ₹1,850 at BigBazaar"),
        ("Razorpay", "Payment of ₹999 confirmed for your subscription"),
        ("GPay", "Rahul collected ₹300 from you for dinner"),
        ("HDFC Bank", "Auto-debit of ₹499 for Netflix subscription"),
        ("Paytm", "Your FASTag was recharged with ₹500"),
        # CRITICAL: OTP messages (high urgency + importance)
        ("ICICI Bank", "OTP: 482931 for transaction of ₹2,500"),
        ("HDFC Bank", "OTP 729312. Valid for 5 minutes. Do NOT share with anyone."),
        ("SBI Bank", "Your OTP is 384920. Use it within 10 minutes. Never share OTP."),
        ("Axis Bank", "OTP: 190283. Valid for 10 mins for UPI transaction. Do not share."),
        ("PhonePe", "OTP: 847291. Valid for 10 minutes. DO NOT share with anyone."),
        ("Paytm", "Your Paytm OTP is 392817. Do not share this with anyone."),
        ("GPay", "OTP 748291 for Google Pay transaction. Don't share this code."),
        ("HDFC Bank", "OTP: 293847. Do not share this with anyone, including bank staff."),
        # CRITICAL: Payment failures (high urgency + importance)
        ("PhonePe", "⚠️ Payment of ₹799 to Swiggy FAILED. Try again."),
        ("GPay", "Transaction declined. Please check your bank account."),
        ("HDFC Bank", "ALERT: Your UPI transaction of ₹5,000 has FAILED"),
        ("Paytm", "Transaction FAILED: ₹1,299 to Amazon. Amount not debited."),
        ("ICICI Bank", "Payment declined. Please retry or use a different payment method."),
        ("SBI Bank", "Your payment of ₹3,500 to Flipkart has failed. Please retry."),
        ("Axis Bank", "FAILED: UPI payment ₹2,000 to Swiggy. Check account balance."),
        ("PhonePe", "Payment failed due to insufficient funds. Add money and retry."),
        # CRITICAL: Suspicious/fraud alerts
        ("HDFC Bank", "Suspicious activity detected on your account. Please verify immediately."),
        ("SBI Bank", "ALERT: Unauthorized access attempt on your account."),
        ("ICICI Bank", "Your debit card was used for an unusual transaction. Verify now."),
        ("Paytm", "Security alert: Login from new device. Please verify it's you."),
    ],
    "delivery": [
        # Food delivery status — ETA / in-progress (urgency 5-6)
        ("Swiggy", "Your order is out for delivery! ETA: 12 mins"),
        ("Swiggy", "Your order has been picked up by the delivery partner"),
        ("Swiggy", "Mihail is on the way with your order"),
        ("Swiggy", "Your order is almost there — ETA: 5 mins"),
        ("Zomato", "Yay! Your food is on the way"),
        ("Zomato", "Your rider is 5 mins away"),
        ("Zomato", "Your order has been picked up!"),
        ("Amazon", "Out for delivery: Noise Cancelling Headphones"),
        ("Amazon", "Your package is nearby, arriving soon"),
        ("Flipkart", "Out for delivery: expected by 6pm today"),
        ("Dunzo", "Your items are packed and on the way!"),
        ("BlueDart", "Your package is out for delivery today"),
        ("Ekart", "Your order will be delivered between 2-4pm today"),
        ("Blinkit", "Out for delivery — arriving in 10 minutes!"),
        ("Uber Eats", "Arjun is heading to your location with your order"),
        # Time-critical: delivery partner arrived (urgency 9-10) — must step outside NOW
        ("Swiggy", "Your delivery partner Ravi has arrived! Step outside to collect your order. 🛵"),
        ("Swiggy", "Delivery partner is at your door. Please collect your order now!"),
        ("Swiggy", "Suresh has arrived with your order! Please step outside now 🏃"),
        ("Swiggy", "Your delivery partner is waiting outside. Come collect NOW!"),
        ("Swiggy", "Ravi is outside your building. Please come down to collect! ⏱"),
        ("Zomato", "Your delivery partner Suresh has arrived! Come collect your food now 🏃"),
        ("Zomato", "Delivery partner is at your door — please come out immediately!"),
        ("Zomato", "Kumar has arrived! Your food will get cold — collect now!"),
        ("Uber", "Your driver Rakesh is outside. Please come down now."),
        ("Uber", "Pradeep has arrived and is waiting outside for you."),
        ("Uber", "Your cab is here! Driver is waiting — please come out."),
        ("Uber Eats", "Delivery partner has arrived at your location!"),
        ("Uber Eats", "Your delivery partner is outside! Come collect your order now."),
        ("Dunzo", "Your delivery partner is at your door — come out now!"),
        ("Blinkit", "Your delivery partner has arrived. Come collect ASAP! ⚡"),
        ("Blinkit", "Rider is at your door! Please step outside immediately."),
        ("Amazon", "Delivery agent Rajan is outside. Please come collect your package!"),
        # Delivered — informational (urgency 4)
        ("Swiggy", "Your order was delivered. Enjoy your meal!"),
        ("Zomato", "Order delivered! Hope you enjoy your meal 🍽️"),
        ("Amazon", "Your order has been delivered. Rate your experience"),
        ("Amazon", "Delivered! Your package was left at the door. 📦"),
        ("Dunzo", "Delivered! Your order from D-Mart is here"),
        ("Flipkart", "Your order has been delivered successfully"),
        ("Blinkit", "Delivered! Your order has been left at the door."),
        # Shipped / tracking updates (urgency 4)
        ("Amazon", "Your package will arrive today"),
        ("Flipkart", "Your order has been shipped — track here"),
        ("DTDC", "Your shipment arrived at the local hub"),
        ("Delhivery", "Shipment arrived at your city hub"),
        ("Amazon", "Delivery attempted — will retry tomorrow"),
    ],
    "system": [
        ("Android", "USB debugging connected"),
        ("System", "Low battery: 15% remaining"),
        ("Android", "App update available: WhatsApp 23.10.1"),
        ("System", "Storage almost full: 512 MB remaining"),
        ("Android", "Bluetooth connected: JBL Headphones"),
        ("System", "Screen time summary: 4h 23m today"),
        ("Google Play", "2 apps updated in the background"),
        ("System", "Wi-Fi network saved: Home_5G"),
        ("Android", "Accessibility service enabled"),
        ("System", "Do Not Disturb turned off"),
        ("Android", "App crashed: Google Maps stopped"),
        ("Google Play Protect", "No harmful apps found"),
        ("System", "Backup completed successfully"),
        ("Google", "Find My Device: your phone's location was shared"),
        ("Android Auto", "Android Auto connected"),
    ],
    "reminder": [
        ("Calendar", "Team standup in 15 minutes"),
        ("Google Calendar", "Doctor appointment at 3pm today"),
        ("Calendar", "Mom's birthday is tomorrow — don't forget!"),
        ("Reminders", "Pay electricity bill by Friday"),
        ("Calendar", "Flight to Bangalore at 6:30am tomorrow"),
        ("Reminders", "Call insurance company about renewal"),
        ("Calendar", "Gym session scheduled: 7am"),
        ("Google Calendar", "Project deadline: submit by 5pm"),
        ("Reminders", "Take medication — 8pm"),
        ("Calendar", "Team lunch at 1pm — Conference Room B"),
        ("Google Calendar", "Ananya's birthday today 🎂"),
        ("Calendar", "Interview scheduled: 2pm — don't be late"),
        ("Reminders", "Water plants"),
        ("Calendar", "Anniversary dinner reservation at 7pm"),
        ("Google Tasks", "Submit assignment before midnight"),
        ("Calendar", "Weekly review — Sunday 10am"),
        ("Reminders", "Renew vehicle insurance by month end"),
        ("Google Calendar", "Workshop starts in 30 mins"),
    ],
}

# ─────────────────────── Tier feature distributions ─────────────────────────

_TIERS = {
    1: {"open_rate": (0.90, 0.99), "resp_time": (10, 120), "freq": (3, 15), "recency": (0.0, 2.0)},
    2: {"open_rate": (0.75, 0.90), "resp_time": (60, 300), "freq": (1, 8), "recency": (0.1, 5.0)},
    3: {"open_rate": (0.60, 0.80), "resp_time": (300, 1800), "freq": (0.5, 5), "recency": (0.5, 10.0)},
    4: {"open_rate": (0.30, 0.50), "resp_time": (1800, 7200), "freq": (0.1, 2), "recency": (1.0, 20.0)},
    5: {"open_rate": (0.02, 0.15), "resp_time": (7200, 86400), "freq": (0.5, 10), "recency": (0.0, 3.0)},
}

# Tier base importance: T1=9, T2=7, T3=6, T4=4, T5=2
_TIER_BASE_IMPORTANCE = {1: 9, 2: 7, 3: 6, 4: 4, 5: 2}

# Category → default urgency base (before context adjustments)
_CAT_URGENCY_BASE = {
    "message": 7,
    "email": 5,
    "social": 3,
    "promo": 1,  # promos are almost never urgent
    "transaction": 7,  # financial events are more time-sensitive than general email
    "system": 3,
    "delivery": 5,
    "reminder": 7,
}

# Urgency keywords that bump up scores — only for personal/work context, NOT promos
_URGENT_KEYWORDS = ["emergency", "urgent", "immediately", "asap", "critical",
                    "help", "danger", "otp", "call me",
                    "are you okay", "it's important", "deadline", "expires",
                    "failed", "declined", "do not share", "verify now",
                    "unauthorized", "unrecognized", "compromised", "unusual login",
                    "suspicious", "security alert", "account locked", "verify your identity"]

# Coordination keywords — real-time planning asks from high-tier senders (T1/T2) warrant urgency boost
# These are NOT urgent in isolation, but become time-sensitive when from someone you know well
_COORDINATION_KEYWORDS = ["when is the meet", "when's the meet", "are you coming",
                           "you coming?", "you joining", "when are you", "where are you",
                           "are you free", "can you come", "meet up", "meetup",
                           "when do we", "shall we", "let's meet", "wanna meet",
                           "coming today", "coming tomorrow", "still on?", "still happening",
                           "confirmed?", "you in?", "join us", "plan for",
                           "when is the plan", "what's the plan", "catch up", "link up"]

# Delivery-specific time-critical keywords (high urgency: partner at door)
_DELIVERY_ARRIVED_KEYWORDS = ["has arrived", "is outside", "is at your door",
                               "waiting outside", "come collect", "step outside",
                               "come out now", "collect asap", "come down now",
                               "at your location", "waiting for you", "please come out",
                               "collect now", "come outside"]

# Transaction critical keywords — payment failure or OTP (high importance regardless of tier)
_TRANSACTION_CRITICAL_KEYWORDS = ["failed", "otp", "declined", "do not share",
                                    "blocked", "suspicious", "unauthorized", "fraud"]

# Security alert keywords — apply to ALL categories including social
_SECURITY_ALERT_KEYWORDS = ["unrecognized device", "unauthorized access", "unusual login",
                              "account was accessed", "account may have been compromised",
                              "security alert", "verify your identity", "account locked",
                              "suspicious activity", "someone tried to access",
                              "unusual login attempt", "new sign-in", "new signin",
                              "was this you", "did you sign in", "did you log in",
                              "login attempt", "sign-in attempt", "unrecognized sign",
                              "change your password", "password was changed",
                              "new device", "unfamiliar device", "unfamiliar location"]

# Promo-style urgency words — these look urgent but should NOT bump scores for promo/shopping
_PROMO_FAKE_URGENCY = ["sale", "off", "free", "deal", "discount", "hurry",
                        "limited", "midnight", "tonight", "offer", "flash", "cashback",
                        "today only", "save", "scratch card", "promo", "code"]

# ──────────────────── Real notification examples ────────────────────────────
# Each entry: title, text, package_name, app_category, label_category,
#             tier, is_group, override_urgency (None=auto), override_importance (None=auto)
_REAL_EXAMPLES = [
    # ── T1 emergency messages (bypass even at 2AM — high importance + urgency forced) ──
    ("Mom", "Emergency! Are you okay? Call me RIGHT NOW", "com.whatsapp", "messaging", "message", 1, False, 10, 10),
    ("Dad", "Please call immediately, it's urgent", "com.whatsapp", "messaging", "message", 1, False, 9, 9),
    ("Partner", "Are you safe? Something happened, call me now!", "com.whatsapp", "messaging", "message", 1, False, 9, 9),
    ("Best Friend", "Emergency! I need help right now. Call me!", "com.whatsapp", "messaging", "message", 1, False, 10, 10),
    ("Mom", "Beta please pick up, it's very important", "com.whatsapp", "messaging", "message", 1, False, 9, 9),
    ("Dad", "Call me ASAP, there's been an accident", "com.whatsapp", "messaging", "message", 1, False, 10, 10),
    ("Partner", "I'm in danger, please call now!!", "com.whatsapp", "messaging", "message", 1, False, 10, 10),
    ("Sister", "Emergency at home! Please come back immediately", "com.whatsapp", "messaging", "message", 1, False, 9, 9),
    # ── Coordination messages — T1/T2: meet/plan asks need a reply (urgency 7-8) ──
    ("Shiva", "When is the meet", "com.whatsapp", "messaging", "message", 2, False, 7, 7),
    ("Rahul", "Are you coming today?", "com.whatsapp", "messaging", "message", 2, False, 7, 7),
    ("Priya", "When are you reaching? We're all here", "com.whatsapp", "messaging", "message", 1, False, 8, 8),
    ("Partner", "Are you free this evening? Let's plan something", "com.whatsapp", "messaging", "message", 1, False, 7, 7),
    ("Ankit", "Still on for tonight?", "com.whatsapp", "messaging", "message", 2, False, 7, 7),
    ("Neha", "Shall we meet at 7? Confirm?", "com.whatsapp", "messaging", "message", 2, False, 8, 7),
    ("Vikram", "You joining us or not? Let us know", "com.whatsapp", "messaging", "message", 2, False, 7, 7),
    ("Mom", "When are you coming home?", "com.whatsapp", "messaging", "message", 1, False, 7, 7),
    ("Best Friend", "You in for Saturday? Making plans", "com.whatsapp", "messaging", "message", 1, False, 7, 8),
    ("Kapil", "Catch up this weekend?", "com.whatsapp", "messaging", "message", 2, False, 6, 6),
    # T3 coordination — still real plans, slightly less urgent (6)
    ("Arjun", "Hey are you coming to the meetup on Friday?", "com.whatsapp", "messaging", "message", 3, False, 6, 6),
    ("Sneha", "When is the plan? Are you free Sunday?", "com.whatsapp", "messaging", "message", 3, False, 6, 6),
    # T4/T5 coordination — acquaintances asking about meetup (lower priority, 4-5)
    ("Ravi", "Are you coming to the office meet?", "com.whatsapp", "messaging", "message", 4, False, 5, 4),
    ("Unknown", "Hey, you joining the event tomorrow?", "com.whatsapp", "messaging", "message", 5, False, 4, 3),
    # ── WhatsApp personal ──
    ("Rahul Sharma", "Good morning! ☀️ Hope you have a great day ahead", "com.whatsapp", "messaging", "message", 2, False, None, None),
    ("Priya", "Heyy! Are you free tonight? 😊", "com.whatsapp", "messaging", "message", 2, False, None, None),
    ("Mom", "Beta, call me when you're free 🙏", "com.whatsapp", "messaging", "message", 1, False, None, None),
    ("Ananya", "Good morning! 🌞 Rise and shine! Hope today is amazing for you ✨", "com.whatsapp", "messaging", "message", 2, False, None, None),
    ("Rohan", "Happy birthday bro!! 🎂🎉 Have a blast!", "com.whatsapp", "messaging", "message", 2, False, None, None),
    ("Neha Verma", "Happy Anniversary! 🎊❤️ May your love grow stronger", "com.whatsapp", "messaging", "message", 2, False, None, None),
    ("WhatsApp", "Missed video call from Ananya", "com.whatsapp", "messaging", "message", 1, False, 8, 9),
    ("WhatsApp", "Incoming WhatsApp call - Vikram Nair is calling...", "com.whatsapp", "messaging", "message", 2, False, 9, 9),
    ("Neha 💕", "📷 Photo", "com.whatsapp", "messaging", "message", 2, False, None, None),
    # ── WhatsApp group chats ──
    ("Office Group", "Amit: Good morning everyone! Meeting at 10am today", "com.whatsapp", "messaging", "message", 3, True, None, None),
    ("College Friends 🎓", "Sneha: Guys check this meme 😂😂", "com.whatsapp", "messaging", "message", 3, True, None, None),
    ("Meme Lords 😂", "Vivek: LMAOOOO 💀💀 bro send this to everyone", "com.whatsapp", "messaging", "message", 4, True, None, None),
    ("Best Friends 💯", "Anjali: HAPPY BIRTHDAY BESTIE!! 🎂🎉🎊", "com.whatsapp", "messaging", "message", 2, True, None, None),
    ("Family Group", "Mom: Happy birthday beta! 🎂🎉 May you have all the happiness", "com.whatsapp", "messaging", "message", 1, True, None, None),
    ("Family Group", "Mom: Subah ki chai ke saath good morning 🍵☕", "com.whatsapp", "messaging", "message", 1, True, None, None),
    ("Apartment Society", "Water supply will be cut tomorrow 8 AM - 12 PM", "com.whatsapp", "messaging", "message", 4, True, 6, 5),
    ("Work Team", "Sanjay: Can someone review the PR before EOD?", "com.whatsapp", "messaging", "message", 3, True, None, None),
    # ── Slack ──
    ("Amit", "Good morning team! 👋", "com.slack", "productivity", "message", 3, False, None, None),
    ("Priya", "PR is ready for review", "com.slack", "productivity", "message", 3, False, None, None),
    ("Sarah Chen", "@you can you review this PR?", "com.slack", "productivity", "message", 2, False, 7, None),
    ("Rohan Mehta", "Hey, do you have 5 minutes for a quick call?", "com.slack", "productivity", "message", 3, False, None, None),
    ("Vikram", "Server is back up, deploy can proceed now ✅", "com.slack", "productivity", "message", 3, False, None, None),
    ("Karan", "Meeting in 15 mins - dial-in link in calendar", "com.slack", "productivity", "message", 3, False, 7, None),
    ("Ananya", "Reminder - submit timesheets by 5 PM today", "com.slack", "productivity", "message", 3, False, 6, None),
    # ── Gmail work ──
    ("Google Account", "Security alert: New sign-in from Chrome on Windows", "com.google.android.gm", "productivity", "email", 3, False, 8, 8),
    ("HR Team", "Your leave request has been approved ✅", "com.google.android.gm", "productivity", "email", 3, False, 5, 6),
    ("Ananya Verma", "Re: Project proposal - sounds good, let's proceed!", "com.google.android.gm", "productivity", "email", 3, False, None, None),
    ("GitHub", "Review requested: PR #124 - Fix authentication bug", "com.google.android.gm", "productivity", "email", 3, False, 6, None),
    ("Zoom", "Meeting reminder: Standup @ 10:00 AM - Join now", "com.google.android.gm", "productivity", "email", 3, False, 7, None),
    # ── Gmail promo ──
    ("Swiggy", "🍕 Your favourite pizza is waiting! 40% off on your next order", "com.google.android.gm", "productivity", "promo", 5, False, 1, 2),
    ("Flipkart", "Big Billion Days: Up to 80% off! Sale ends midnight", "com.google.android.gm", "productivity", "promo", 5, False, 1, 2),
    # ── Late night promo — should ALWAYS suppress ──
    ("Zomato", "Flash sale tonight only — 60% off! Order before midnight!", "in.swiggy.android", "shopping", "promo", 5, False, 1, 1),
    ("Amazon", "Night deals: hurry, limited stock ends at midnight!", "in.amazon.mShop.android.shopping", "shopping", "promo", 5, False, 1, 1),
    ("Swiggy", "Midnight munchies? Order now with FREE delivery!", "in.swiggy.android", "shopping", "promo", 5, False, 1, 1),
    ("Myntra", "Night out sale: last chance tonight only!", "com.myntra.android", "shopping", "promo", 5, False, 1, 1),
    ("Flipkart", "Deal ending in 2 hours — don't miss out!", "com.flipkart.android", "shopping", "promo", 5, False, 1, 1),
    ("MakeMyTrip", "Special midnight offer on flights — book now!", "com.makemytrip", "shopping", "promo", 5, False, 1, 1),
    # ── Instagram ──
    ("Instagram", "rahul_sharma23 liked your photo.", "com.instagram.android", "social", "social", 4, False, None, None),
    ("Instagram", "priya_travels started following you.", "com.instagram.android", "social", "social", 4, False, None, None),
    ("Instagram", "sneha_foodie commented: Omg this is stunning! 😍", "com.instagram.android", "social", "social", 3, False, None, None),
    ("Instagram", "anjali_design mentioned you in a comment: @you this reminded me of you!", "com.instagram.android", "social", "social", 3, False, None, None),
    ("Instagram", "sneha_foodie sent you a message.", "com.instagram.android", "social", "message", 3, False, None, None),
    ("Instagram", "karan_memes sent you a reel: This is literally you 😂", "com.instagram.android", "social", "message", 3, False, None, None),
    ("Instagram", "karan_99: Good morning!! check my story 🌅", "com.instagram.android", "social", "message", 3, False, None, None),
    ("Instagram", "Your story was seen by 234 people", "com.instagram.android", "social", "social", 5, False, None, None),
    # ── Facebook / Messenger ──
    ("Facebook", "Rohan Mehta commented: Great photo!", "com.facebook.katana", "social", "social", 4, False, None, None),
    ("Facebook", "Priya Sharma and 12 others reacted to your photo.", "com.facebook.katana", "social", "social", 4, False, None, None),
    ("Facebook", "Amit Kumar tagged you in a photo.", "com.facebook.katana", "social", "social", 4, False, None, None),
    ("Facebook", "Today is Sneha Patel's birthday! 🎂", "com.facebook.katana", "social", "social", 4, False, 3, 3),
    ("Facebook", "Vikram Singh invited you to: New Year Party 2026 🎉", "com.facebook.katana", "social", "social", 3, False, None, None),
    ("Neha Gupta", "Hey! Are you coming to the meet?", "com.facebook.orca", "messaging", "message", 3, False, None, None),
    # ── Snapchat ──
    ("Snapchat", "Priya Verma sent you a Snap!", "com.snapchat.android", "messaging", "message", 3, False, None, None),
    ("Snapchat", "rohit_snap sent you a Chat", "com.snapchat.android", "messaging", "message", 3, False, None, None),
    ("Snapchat", "Dev_snaps is typing...", "com.snapchat.android", "messaging", "message", 3, False, 3, 4),
    ("Snapchat", "Your Snap to kavya_snaps was opened 😮", "com.snapchat.android", "messaging", "social", 4, False, 2, 3),
    ("Snapchat", "Neha took a screenshot of your Snap!", "com.snapchat.android", "messaging", "social", 4, False, 4, 5),
    ("Snapchat", "Don't let your Snapstreak with Karan die! 💨", "com.snapchat.android", "messaging", "social", 4, False, 3, 3),
    ("Snapchat", "🔥 Your Snapstreak with priya_snaps will end soon!", "com.snapchat.android", "messaging", "social", 4, False, 3, 3),
    # ── Swiggy delivery ──
    ("Swiggy", "Your order from McDonald's is confirmed! Getting it ready for you 🍔", "in.swiggy.android", "shopping", "delivery", 5, False, 4, 4),
    ("Swiggy", "Your Delivery Partner Ravi has picked up your order 🛵", "in.swiggy.android", "shopping", "delivery", 5, False, 5, 5),
    ("Swiggy", "Your order is out for delivery! ETA: 12 mins 🚀", "in.swiggy.android", "shopping", "delivery", 5, False, 7, 6),
    ("Swiggy", "Your delivery partner Suresh has arrived! Step outside to collect your order. 🛵", "in.swiggy.android", "shopping", "delivery", 5, False, 9, 9),
    ("Swiggy", "Your delivery partner Ravi is outside! Come collect your food now.", "in.swiggy.android", "shopping", "delivery", 5, False, 9, 9),
    ("Swiggy", "Delivery partner waiting outside — please step outside now!", "in.swiggy.android", "shopping", "delivery", 5, False, 9, 9),
    ("Swiggy", "Order Delivered! ✅ Hope you enjoy your meal.", "in.swiggy.android", "shopping", "delivery", 5, False, 4, 5),
    ("Swiggy", "Yikes! Your order is slightly delayed. New ETA: 35 mins.", "in.swiggy.android", "shopping", "delivery", 5, False, 5, 5),
    # ── Swiggy promo ──
    ("Swiggy", "🔥 Lunchtime offer: Get ₹100 off on orders above ₹299!", "in.swiggy.android", "shopping", "promo", 5, False, 1, 2),
    ("Swiggy Instamart", "Running low on groceries? Get delivery in 10 mins!", "in.swiggy.android", "shopping", "promo", 5, False, 1, 2),
    ("Swiggy Super", "Your Swiggy Super membership expires in 3 days.", "in.swiggy.android", "shopping", "promo", 5, False, 2, 3),
    # ── Uber delivery/ride ──
    ("Uber", "Your driver Rakesh is 3 minutes away 🚗", "com.ubercab", "shopping", "delivery", 5, False, 8, 7),
    ("Uber", "Pradeep has arrived. Your UberGo is outside.", "com.ubercab", "shopping", "delivery", 5, False, 9, 9),
    ("Uber", "Your driver has arrived and is waiting outside. Please come down.", "com.ubercab", "shopping", "delivery", 5, False, 9, 9),
    ("Uber", "Your ride is complete. ₹287 charged to Paytm.", "com.ubercab", "shopping", "transaction", 5, False, 4, 5),
    ("Uber", "Surge pricing is active in your area.", "com.ubercab", "shopping", "system", 5, False, 4, 4),
    ("Uber Eats", "Arjun is picking up your order from Burger King. ETA: 20 mins", "com.ubercab", "shopping", "delivery", 5, False, 5, 5),
    ("Uber Eats", "Your delivery partner has arrived! Please come out to collect your food.", "com.ubercab", "shopping", "delivery", 5, False, 9, 9),
    # ── Uber promo ──
    ("Uber", "20% off your next 3 rides! Use code RIDE20", "com.ubercab", "shopping", "promo", 5, False, 1, 2),
    # ── Zomato delivery arrived ──
    ("Zomato", "Your delivery partner has arrived! Come collect your food now", "com.application.zomato", "shopping", "delivery", 5, False, 9, 9),
    ("Zomato", "Delivery partner is at your door — please come out immediately!", "com.application.zomato", "shopping", "delivery", 5, False, 9, 9),
    ("Zomato", "Kumar has arrived! Your food will get cold — collect now!", "com.application.zomato", "shopping", "delivery", 5, False, 9, 9),
    ("Zomato", "Your delivery partner Suresh has arrived! Step outside to collect your order.", "com.application.zomato", "shopping", "delivery", 5, False, 9, 9),
    # ── Blinkit delivery arrived ──
    ("Blinkit", "Your delivery partner has arrived. Come collect ASAP!", "com.grofers.consumer", "shopping", "delivery", 5, False, 9, 9),
    ("Blinkit", "Rider is at your door! Please step outside immediately.", "com.grofers.consumer", "shopping", "delivery", 5, False, 9, 9),
    # ── Dunzo delivery arrived ──
    ("Dunzo", "Your delivery partner is at your door — come out now!", "com.dunzo.user", "shopping", "delivery", 5, False, 9, 9),
    ("Dunzo", "Delivery partner waiting outside. Please step outside to collect your order.", "com.dunzo.user", "shopping", "delivery", 5, False, 9, 9),
    ("Dunzo", "Your Dunzo delivery has arrived! Step outside to collect.", "com.dunzo.user", "shopping", "delivery", 5, False, 9, 9),
    ("Dunzo", "Partner is at your door. Collect now or your order may be returned.", "com.dunzo.user", "shopping", "delivery", 5, False, 9, 9),
    # ── Amazon delivery ──
    ("Amazon", "Your order of boAt Rockerz 255 has been shipped!", "in.amazon.mShop.android.shopping", "shopping", "delivery", 5, False, 4, 5),
    ("Amazon", "Out for delivery: Sony WH-1000XM5. Today by 8 PM.", "in.amazon.mShop.android.shopping", "shopping", "delivery", 5, False, 6, 6),
    ("Amazon", "Delivered! Your package was left at the door. 📦", "in.amazon.mShop.android.shopping", "shopping", "delivery", 5, False, 4, 5),
    # ── Amazon transaction ──
    ("Amazon Pay", "Payment of ₹1,299 to Amazon successful.", "in.amazon.mShop.android.shopping", "finance", "transaction", 5, False, 5, 5),
    # ── Amazon promo ──
    ("Amazon", "Lightning Deal ends in 30 mins: JBL Bluetooth Speaker — ₹1,999", "in.amazon.mShop.android.shopping", "shopping", "promo", 5, False, 1, 2),
    ("Amazon", "Great Indian Festival starts now! Up to 70% off 🎉", "in.amazon.mShop.android.shopping", "shopping", "promo", 5, False, 1, 2),
    ("Amazon", "Price drop! The item in your Saved list dropped by ₹500.", "in.amazon.mShop.android.shopping", "shopping", "promo", 5, False, 2, 3),
    # ── PhonePe transactions ──
    ("PhonePe", "₹500 paid to Rahul Kumar. UPI Ref ✅", "com.phonepe.app", "finance", "transaction", 5, False, 5, 6),
    ("PhonePe", "You received ₹1,200 from Priya Sharma.", "com.phonepe.app", "finance", "transaction", 5, False, 5, 6),
    ("PhonePe", "Vivek Reddy sent you a payment request of ₹350", "com.phonepe.app", "finance", "transaction", 5, False, 6, 6),
    ("PhonePe", "⚠️ Payment of ₹799 to Swiggy FAILED. Try again.", "com.phonepe.app", "finance", "transaction", 5, False, 9, 9),
    ("PhonePe", "OTP: 847291. Valid for 10 minutes. DO NOT share with anyone.", "com.phonepe.app", "finance", "transaction", 5, False, 10, 9),
    ("PhonePe", "Electricity bill of ₹892 paid successfully.", "com.phonepe.app", "finance", "transaction", 5, False, 4, 5),
    ("PhonePe", "Your FASTag wallet balance is low (₹50). Recharge now.", "com.phonepe.app", "finance", "transaction", 5, False, 7, 6),
    # ── PhonePe promo ──
    ("PhonePe", "🎁 Scratch card unlocked! You won ₹30 cashback", "com.phonepe.app", "finance", "promo", 5, False, 1, 2),
    # ── GPay transactions ──
    ("Google Pay", "₹2,000 sent to Rohan Sharma successfully 🎉", "com.google.android.apps.nbu.paisa.user", "finance", "transaction", 5, False, 5, 6),
    ("Google Pay", "You received ₹500 from Sneha Joshi.", "com.google.android.apps.nbu.paisa.user", "finance", "transaction", 5, False, 5, 6),
    ("Google Pay", "Amit requested ₹600 from you. Pay now?", "com.google.android.apps.nbu.paisa.user", "finance", "transaction", 5, False, 6, 6),
    ("Google Pay", "Transaction declined. Please check your bank account.", "com.google.android.apps.nbu.paisa.user", "finance", "transaction", 5, False, 9, 9),
    ("Google Pay", "Electricity bill of ₹1,847 paid to BESCOM ✅", "com.google.android.apps.nbu.paisa.user", "finance", "transaction", 5, False, 4, 5),
    # ── GPay promo ──
    ("Google Pay", "Win rewards! Pay with Google Pay at any store", "com.google.android.apps.nbu.paisa.user", "finance", "promo", 5, False, 1, 2),
    # ── LinkedIn ──
    ("LinkedIn", "Priya Sharma viewed your profile", "com.linkedin.android", "social", "social", 5, False, None, None),
    ("LinkedIn", "Your post is trending! 🔥", "com.linkedin.android", "social", "social", 4, False, None, None),
    ("LinkedIn", "Rahul Kumar sent you a connection request", "com.linkedin.android", "social", "social", 4, False, None, None),
    ("LinkedIn", "Amit Verma commented: Really insightful, thanks for sharing!", "com.linkedin.android", "social", "social", 4, False, None, None),
    ("LinkedIn", "HR from Flipkart sent you an InMail: Exciting opportunity!", "com.linkedin.android", "social", "social", 3, False, 5, 5),
    ("LinkedIn", "New job: Senior Software Engineer at Google — Bangalore", "com.linkedin.android", "social", "social", 5, False, 3, 3),
    ("LinkedIn", "Today is Sneha Patel's work anniversary! Wish her 🎊", "com.linkedin.android", "social", "social", 5, False, 2, 2),
    # ── Twitter/X ──
    ("Twitter", "techguru_india retweeted your post.", "com.twitter.android", "social", "social", 5, False, None, None),
    ("Twitter", "Neha Sharma sent you a Direct Message.", "com.twitter.android", "social", "message", 3, False, 5, 5),
    ("X (Twitter)", "Trending in India: #HappyDiwali — 2.3M tweets", "com.twitter.android", "social", "social", 5, False, 2, 2),
    ("Twitter", "Your account has a new login from a new device.", "com.twitter.android", "social", "social", 5, False, 8, 9),
    ("Twitter", "Your account may have been compromised. Change your password now.", "com.twitter.android", "social", "social", 5, False, 9, 10),
    # ── Reddit ──
    ("Reddit", "u/rahul_irl sent you a Direct Message.", "com.reddit.frontpage", "social", "message", 3, False, 5, 5),
    ("Reddit", "u/mod_india removed your post.", "com.reddit.frontpage", "social", "social", 4, False, 4, 5),
    # ── Discord DMs and @mentions ──
    ("rahul_dev", "yo are you around?", "com.discord", "messaging", "message", 3, False, 6, 7),
    ("Priya", "Can you jump on voice for 5 mins?", "com.discord", "messaging", "message", 2, False, 7, 8),
    ("Discord", "You were mentioned in #general: @you check this PR", "com.discord", "messaging", "message", 3, False, 6, 7),
    ("Discord", "@everyone Exam results drop tonight at 10 PM!", "com.discord", "messaging", "social", 4, False, 5, 6),
    ("sneha_dev", "sneha_dev started a call in Study Group", "com.discord", "messaging", "social", 3, False, 5, 5),
    # ── LinkedIn InMail (recruiter DMs — high importance) ──
    ("LinkedIn", "HR from Google sent you an InMail: Exciting opportunity — SDE3", "com.linkedin.android", "social", "social", 3, False, 7, 9),
    ("LinkedIn", "Flipkart recruiter sent you a message: Are you open to new roles?", "com.linkedin.android", "social", "social", 3, False, 6, 8),
    ("LinkedIn", "Your application was viewed by the recruiter at Meesho.", "com.linkedin.android", "social", "social", 4, False, 4, 6),
    # ── Security alerts from social apps (critical — MUST bypass DND) ──
    ("Instagram", "A login from an unrecognized device was detected.", "com.instagram.android", "social", "social", 4, False, 9, 9),
    ("Instagram", "Your account will be restricted unless you verify your identity.", "com.instagram.android", "social", "social", 4, False, 9, 9),
    ("Instagram", "Unusual login: we detected a sign-in from a new location.", "com.instagram.android", "social", "social", 4, False, 9, 9),
    ("Facebook", "We noticed an unusual login attempt on your account.", "com.facebook.katana", "social", "social", 4, False, 9, 10),
    ("Facebook", "Security alert: someone tried to access your account.", "com.facebook.katana", "social", "social", 4, False, 9, 10),
    ("Twitter", "Your account has a new login from a new device.", "com.twitter.android", "social", "social", 4, False, 9, 9),
    ("Twitter", "Your account may have been compromised. Change your password now.", "com.twitter.android", "social", "social", 4, False, 10, 10),
    ("LinkedIn", "Security alert: your account was accessed from a new device.", "com.linkedin.android", "social", "social", 4, False, 9, 9),
    ("Google", "Security alert: New sign-in from Chrome on Windows", "com.google.android.gm", "productivity", "email", 3, False, 9, 9),
    ("Google", "A new sign-in on Windows. Was this you?", "com.google.android.gm", "productivity", "email", 3, False, 9, 9),
    ("Google", "New sign-in to your Google Account from an unrecognized device.", "com.google.android.gm", "productivity", "email", 3, False, 9, 9),
    # Google security alerts appearing in social/notification shade (app_cat=social context)
    # Tier 4 (moderate signals) — model must learn: security text overrides sender open_rate
    ("Google", "A new sign-in on Windows. Was this you?", "com.google.android.googlequicksearchbox", "social", "social", 4, False, 9, 9),
    ("Google", "Suspicious sign-in prevented. Review activity on your account.", "com.google.android.googlequicksearchbox", "social", "social", 4, False, 9, 9),
    # Tier 5 (very low open_rate) — security alerts must bypass DND regardless of sender history
    ("Google", "New device sign-in detected. Was this you?", "com.google.android.googlequicksearchbox", "social", "social", 5, False, 9, 9),
    ("Google", "A new sign-in on Windows. Was this you?", "com.google.android.googlequicksearchbox", "social", "social", 5, False, 9, 9),
    ("Google", "Someone signed in to your account from a new device.", "com.google.android.googlequicksearchbox", "social", "social", 5, False, 9, 9),
    ("Facebook", "A new sign-in on iPhone. Was this you?", "com.facebook.katana", "social", "social", 4, False, 9, 9),
    ("Facebook", "An unusual login attempt was made on your account.", "com.facebook.katana", "social", "social", 4, False, 9, 10),
    ("Facebook", "We noticed an unusual login attempt. Was this you?", "com.facebook.katana", "social", "social", 5, False, 9, 9),
    ("Facebook", "Unusual login activity detected on your account.", "com.facebook.katana", "social", "social", 5, False, 9, 9),
    # WhatsApp security alerts — messaging app_cat, must bypass with T5 signals
    ("WhatsApp", "Someone tried to access your account. Verify your identity.", "com.whatsapp", "messaging", "message", 4, False, 9, 9),
    ("WhatsApp", "Your account was accessed from a new device.", "com.whatsapp", "messaging", "message", 4, False, 9, 9),
    ("WhatsApp", "New sign-in detected. Was this you? If not, secure your account.", "com.whatsapp", "messaging", "message", 4, False, 9, 9),
    ("WhatsApp", "Someone tried to access your account. Verify your identity.", "com.whatsapp", "messaging", "message", 5, False, 9, 9),
    ("WhatsApp", "A new sign-in was detected on your WhatsApp account.", "com.whatsapp", "messaging", "message", 5, False, 9, 9),
    ("WhatsApp", "Verify your identity to keep your account secure.", "com.whatsapp", "messaging", "message", 5, False, 9, 9),
    # ── Snapchat streak warnings (fake urgency — look urgent, low real importance) ──
    ("Snapchat", "🔥 Your Snapstreak with Priya will end in 1 hour!", "com.snapchat.android", "messaging", "social", 4, False, 3, 3),
    ("Snapchat", "Don't let your Snapstreak with Karan die! 💨", "com.snapchat.android", "messaging", "social", 4, False, 3, 3),
    # ── YouTube ──
    ("YouTube", "Techburner just uploaded: iPhone 18 Pro Unboxing - INSANE camera!", "com.google.android.youtube", "media", "social", 5, False, None, None),
    ("YouTube", "CarryMinati is LIVE now! Join the stream 🔴", "com.google.android.youtube", "media", "social", 4, False, 4, 4),
    ("YouTube", "Priya commented on your video: Amazing tutorial, really helped me!", "com.google.android.youtube", "media", "social", 4, False, None, None),
    ("YouTube", "Congrats! Your channel reached 1,000 subscribers 🎉", "com.google.android.youtube", "media", "social", 4, False, 3, 5),
    # ── Google Photos ──
    ("Google Photos", "New memory: See your best photos from Goa trip 🌊", "com.google.android.apps.photos", "media", "social", 5, False, 2, 3),
    ("Google Photos", "Priya Sharma shared an album with you: Diwali 2025 🪔", "com.google.android.apps.photos", "media", "social", 3, False, 3, 4),
    ("Google Photos", "Anjali Sharma commented on a shared photo: This one is my favorite! 😍", "com.google.android.apps.photos", "media", "social", 4, False, 3, 4),
    ("Google Photos", "Your storage is 95% full. Upgrade to Google One.", "com.google.android.apps.photos", "media", "system", 5, False, 5, 5),
    # ── Google Maps ──
    ("Google Maps", "Time to leave for Work! Traffic is heavy, leave now.", "com.google.android.apps.maps", "system", "system", 5, False, 8, 7),
    ("Google Maps", "Faster route available. Save 8 minutes via NH48.", "com.google.android.apps.maps", "system", "system", 5, False, 5, 5),
    ("Google Maps", "Live traffic: Accident reported near Silk Board Junction. Expect 35 min delay.", "com.google.android.apps.maps", "system", "system", 5, False, 6, 5),
    ("Google Maps", "You have arrived at your destination: Indiranagar, Bangalore 📍", "com.google.android.apps.maps", "system", "system", 5, False, 3, 3),
    ("Google Maps", "Priya Sharma shared their location with you.", "com.google.android.apps.maps", "system", "system", 3, False, 4, 5),
]


def _has_urgency_keywords(text: str, category: str = "") -> bool:
    t = text.lower()
    if category in ("promo", "social"):
        return False  # promo/social "urgency" words don't bump scores
    return any(kw in t for kw in _URGENT_KEYWORDS)


def _sample_tier_features(rng: np.random.Generator, tier: int):
    d = _TIERS[tier]
    open_rate = rng.uniform(*d["open_rate"])
    resp_time = rng.uniform(*d["resp_time"])
    freq = rng.uniform(*d["freq"])
    recency = rng.uniform(*d["recency"])
    conv_ratio = rng.uniform(0.3, 0.9) if tier <= 2 else rng.uniform(0.0, 0.3)
    return open_rate, resp_time, freq, recency, conv_ratio


def generate_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cat_order = ["message", "email", "social", "promo", "transaction", "system", "delivery", "reminder"]
    cat_weight_map = {"message": 0.30, "social": 0.15, "promo": 0.15, "transaction": 0.10,
                      "email": 0.10, "system": 0.07, "delivery": 0.08, "reminder": 0.05}

    app_pool_by_cat = {}
    for pkg, name, cat, personal in _APPS:
        app_pool_by_cat.setdefault(cat, []).append((pkg, name, personal))

    app_cats = ["messaging", "social", "shopping", "finance", "productivity", "media", "system"]

    rows = []
    for _ in range(n_samples):
        # Pick category
        cat = rng.choice(cat_order, p=[cat_weight_map[c] for c in cat_order])

        # Pick app matching category (with fallback)
        if cat == "message":
            app_cat = "messaging"
        elif cat == "email":
            app_cat = "productivity"
        elif cat == "social":
            app_cat = "social"
        elif cat in ("promo", "delivery"):
            app_cat = "shopping"
        elif cat == "transaction":
            app_cat = "finance"
        elif cat == "system":
            app_cat = "system"
        elif cat == "reminder":
            app_cat = "productivity"
        else:
            app_cat = "messaging"

        apps_for_cat = app_pool_by_cat.get(app_cat, app_pool_by_cat["messaging"])
        pkg, app_name, can_be_personal = apps_for_cat[int(rng.integers(len(apps_for_cat)))]

        # Pick relationship tier (must happen before template selection for messages)
        if cat in ("promo",):
            tier = 5
        elif cat in ("system",):
            tier = rng.choice([4, 5], p=[0.5, 0.5])
        elif cat in ("transaction",):
            tier = rng.choice([3, 4, 5], p=[0.4, 0.4, 0.2])
        elif cat in ("delivery",):
            tier = rng.choice([4, 5], p=[0.6, 0.4])
        elif cat in ("message",):
            tier = rng.choice([1, 2, 3, 4, 5], p=[0.20, 0.30, 0.20, 0.15, 0.15])
        elif cat in ("email",):
            tier = rng.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
        elif cat in ("social",):
            tier = rng.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
        elif cat in ("reminder",):
            tier = rng.choice([2, 3], p=[0.5, 0.5])
        else:
            tier = int(rng.integers(1, 6))

        # Pick text from templates
        # For message category: T1/T2 must not draw spam/cold-outreach templates
        # (spam templates mixed with high-tier labels inverts labels: insurance spam @ T1 → imp=10)
        templates = _TEXTS[cat]
        if cat == "message" and tier in (1, 2):
            _SPAM_SENDERS = {"Insurance Agent", "Unknown", "Marketing", "Recruiter",
                             "Sales", "Unknown Number", "Job Portal", "Agent",
                             "Spam", "MLM Company"}
            templates = [t for t in templates if t[0] not in _SPAM_SENDERS]
        title, body = templates[int(rng.integers(len(templates)))]
        has_urgent = _has_urgency_keywords(title + " " + body, category=cat)

        # Cold start: ~15% of samples
        is_cold_start = rng.random() < 0.15

        # Context features
        hour = int(rng.integers(0, 24))
        dow = int(rng.integers(0, 7))
        is_weekend = dow >= 5
        is_sleep = (hour >= 23) or (hour < 7)
        is_group = cat == "message" and rng.random() < 0.3
        has_reply = cat in ("message", "email") and rng.random() < 0.8
        notif_importance_raw = int(rng.choice([2, 3, 4, 5], p=[0.1, 0.3, 0.4, 0.2]))
        is_messaging_style = cat == "message" or app_cat == "messaging"
        flag_fs = cat == "system" and rng.random() < 0.2

        # App-level stats (correlated with tier for messaging apps)
        if is_cold_start:
            s_open, s_resp, s_freq, s_recency, s_conv = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
            app_open = float("nan")
            app_volume = float("nan")
            mins_since = float("nan")
        else:
            s_open, s_resp, s_freq, s_recency, s_conv = _sample_tier_features(rng, tier)
            app_open = float(rng.uniform(0.3, 0.9) if app_cat == "messaging" else rng.uniform(0.05, 0.4))
            app_volume = float(rng.uniform(5, 60) if app_cat in ("shopping", "finance") else rng.uniform(1, 30))
            mins_since = float(rng.choice([float("nan"), rng.uniform(1, 1440)], p=[0.3, 0.7]))

        # Critical transaction flag (OTP, failed payment = always high importance)
        is_critical_transaction = (cat == "transaction" and
            any(kw in (title + " " + body).lower() for kw in _TRANSACTION_CRITICAL_KEYWORDS))

        # Delivery arrived at door = must act immediately (high importance + urgency)
        delivery_arrived = (cat == "delivery" and
            any(kw in (title + " " + body).lower() for kw in _DELIVERY_ARRIVED_KEYWORDS))

        # Security alert — applies to any category including social
        is_security_alert = any(kw in (title + " " + body).lower() for kw in _SECURITY_ALERT_KEYWORDS)

        # Label: importance
        if is_security_alert:
            imp = int(rng.integers(8, 11))  # account security alerts are always critical
        elif is_critical_transaction:
            imp = int(rng.integers(8, 11))  # OTP and failed payments are always important (8-10)
        elif delivery_arrived:
            imp = int(rng.integers(8, 11))  # delivery partner waiting at door (8-10)
        elif is_cold_start:
            if cat == "promo":
                # Promos are always low importance even without behavioral data
                imp = int(rng.integers(1, 3))
            elif has_urgent:
                imp = int(rng.integers(5, 8))
            else:
                imp = int(rng.integers(4, 7))
        else:
            base = _TIER_BASE_IMPORTANCE[tier]
            bonus = 2 if has_urgent else 0
            noise = int(rng.integers(-1, 2))
            # Group chats are less important than direct messages (−2 penalty)
            group_penalty = -2 if is_group else 0
            # T5 messages (spam/unsolicited) get hard cap on importance
            t5_cap = 3 if (tier == 5 and cat == "message") else 10
            imp = int(np.clip(min(base + bonus + noise + group_penalty, t5_cap), 1, 10))

        # Label: urgency (penalize sleep hours for non-critical categories)
        urg_base = _CAT_URGENCY_BASE[cat]
        # T1 direct messages are high urgency (open_rate >= 0.85 implies close relationship)
        tier1_bonus = 2 if (tier == 1 and cat == "message" and not is_group) else 0
        # T5 messages are very low priority — dampen urgency strongly
        tier5_penalty = -4 if (tier == 5 and cat == "message") else 0
        # Delivery partner at door = extremely time-critical (must act immediately)
        delivery_arrived = cat == "delivery" and any(kw in (title + " " + body).lower() for kw in _DELIVERY_ARRIVED_KEYWORDS)
        delivery_arrived_bonus = 5 if delivery_arrived else 0
        # Coordination asks from T1/T2 — "when is the meet", "are you coming" etc. need a reply soon
        has_coordination = (cat == "message" and not is_group and
            any(kw in (title + " " + body).lower() for kw in _COORDINATION_KEYWORDS))
        coordination_bonus = 2 if (has_coordination and tier in (1, 2)) else 0
        urg_bonus = 2 if has_urgent else 0
        urg_noise = int(rng.integers(-1, 2))
        # Security alerts and critical transactions → forced high urgency
        if is_security_alert or is_critical_transaction:
            urg = int(rng.integers(8, 11))  # forced to 8-10
        else:
            urg = int(np.clip(urg_base + tier1_bonus + tier5_penalty + delivery_arrived_bonus + urg_bonus + coordination_bonus + urg_noise, 1, 10))
        # Sleep penalty: promos/social/system at night even lower urgency
        # BUT: T1 urgent messages bypass sleep penalty (close family/partner emergency)
        is_t1_urgent = (tier == 1 and cat == "message" and not is_group and has_urgent)
        if is_sleep and cat in ("promo", "social", "system") and not is_t1_urgent and not is_security_alert:
            urg = int(np.clip(int(urg * 0.4), 1, 10))
        if is_sleep and cat == "promo":
            urg = min(urg, 1)  # sleep + promo → urgency=1 always
        # Hard cap: promos are never urgent regardless of text or time
        if cat == "promo":
            urg = min(urg, 2)

        # One-hot app category
        one_hot = {f"app_cat_{c}": (1 if c == app_cat else 0) for c in app_cats}

        row = {
            **{f"embed_{i}": 0.0 for i in range(128)},  # placeholder; filled in feature engineering
            **one_hot,
            "sender_open_rate": s_open,
            "sender_avg_response_time_sec": s_resp,
            "sender_interaction_frequency": s_freq,
            "sender_recency_days": s_recency,
            "sender_conversation_ratio": s_conv,
            "app_open_rate": app_open,
            "app_daily_volume": app_volume,
            "hour_of_day": float(hour),
            "is_weekend": float(is_weekend),
            "is_sleep_hours": float(is_sleep),
            "minutes_since_last_from_sender": mins_since,
            "is_group_conversation": float(is_group),
            "notification_has_reply_action": float(has_reply),
            "notification_importance_raw": float(notif_importance_raw),
            "is_messaging_style": float(is_messaging_style),
            "flag_foreground_service": float(flag_fs),
            # metadata
            "title": title,
            "text": body,
            "package_name": pkg,
            "app_category": app_cat,
            "label_importance": imp,
            "label_urgency": urg,
            "label_category": cat,
        }
        rows.append(row)

    # Append real notification examples
    for (title, body, pkg, app_cat, cat, tier, is_group,
         override_urg, override_imp) in _REAL_EXAMPLES:
        is_cold_start = rng.random() < 0.10  # lower cold-start rate for real examples
        hour = int(rng.integers(7, 23))
        dow = int(rng.integers(0, 7))
        is_weekend = dow >= 5
        is_sleep = (hour >= 23) or (hour < 7)
        has_urgent = _has_urgency_keywords(title + " " + body, category=cat)
        has_reply = cat in ("message", "email") and rng.random() < 0.8
        notif_importance_raw = int(rng.choice([2, 3, 4, 5], p=[0.1, 0.3, 0.4, 0.2]))
        is_messaging_style = cat == "message" or app_cat == "messaging"
        flag_fs = False

        if is_cold_start:
            s_open, s_resp, s_freq, s_recency, s_conv = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
            app_open = float("nan")
            app_volume = float("nan")
            mins_since = float("nan")
        else:
            s_open, s_resp, s_freq, s_recency, s_conv = _sample_tier_features(rng, tier)
            app_open = float(rng.uniform(0.3, 0.9) if app_cat == "messaging" else rng.uniform(0.05, 0.4))
            app_volume = float(rng.uniform(5, 60) if app_cat in ("shopping", "finance") else rng.uniform(1, 30))
            mins_since = float(rng.choice([float("nan"), rng.uniform(1, 1440)], p=[0.3, 0.7]))

        is_critical_transaction = (cat == "transaction" and
            any(kw in (title + " " + body).lower() for kw in _TRANSACTION_CRITICAL_KEYWORDS))
        delivery_arrived_re = (cat == "delivery" and
            any(kw in (title + " " + body).lower() for kw in _DELIVERY_ARRIVED_KEYWORDS))
        is_security_alert_re = any(kw in (title + " " + body).lower() for kw in _SECURITY_ALERT_KEYWORDS)

        if override_imp is not None:
            # For critical overrides (9-10), apply no noise to avoid pulling down
            if override_imp >= 9:
                imp = int(np.clip(override_imp, 1, 10))
            else:
                imp = int(np.clip(override_imp + int(rng.integers(-1, 2)), 1, 10))
        elif is_security_alert_re:
            imp = int(rng.integers(8, 11))
        elif is_critical_transaction:
            imp = int(rng.integers(8, 11))
        elif delivery_arrived_re:
            imp = int(rng.integers(8, 11))
        elif is_cold_start:
            if cat == "promo":
                imp = int(rng.integers(1, 3))
            else:
                imp = int(rng.integers(4, 7))
        else:
            base = _TIER_BASE_IMPORTANCE[tier]
            bonus = 2 if has_urgent else 0
            noise = int(rng.integers(-1, 2))
            group_penalty = -2 if is_group else 0
            t5_cap = 3 if (tier == 5 and cat == "message") else 10
            imp = int(np.clip(min(base + bonus + noise + group_penalty, t5_cap), 1, 10))

        if override_urg is not None:
            # For critical overrides (9-10), apply no noise
            if override_urg >= 9:
                urg = int(np.clip(override_urg, 1, 10))
            else:
                urg = int(np.clip(override_urg + int(rng.integers(-1, 2)), 1, 10))
        else:
            urg_base = _CAT_URGENCY_BASE[cat]
            tier1_bonus = 2 if (tier == 1 and cat == "message" and not is_group) else 0
            tier5_penalty = -4 if (tier == 5 and cat == "message") else 0
            delivery_arrived = cat == "delivery" and any(kw in (title + " " + body).lower() for kw in _DELIVERY_ARRIVED_KEYWORDS)
            delivery_arrived_bonus = 5 if delivery_arrived else 0
            has_coordination = (cat == "message" and not is_group and
                any(kw in (title + " " + body).lower() for kw in _COORDINATION_KEYWORDS))
            coordination_bonus = 2 if (has_coordination and tier in (1, 2)) else 0
            urg_bonus = 2 if has_urgent else 0
            urg_noise = int(rng.integers(-1, 2))
            if is_security_alert_re or is_critical_transaction:
                urg = int(rng.integers(8, 11))
            else:
                urg = int(np.clip(urg_base + tier1_bonus + tier5_penalty + delivery_arrived_bonus + urg_bonus + coordination_bonus + urg_noise, 1, 10))
            is_t1_urgent = (tier == 1 and cat == "message" and not is_group and has_urgent)
            if is_sleep and cat in ("promo", "social", "system") and not is_t1_urgent:
                urg = int(np.clip(int(urg * 0.4), 1, 10))
            if is_sleep and cat == "promo":
                urg = min(urg, 1)
        if cat == "promo":
            urg = min(urg, 2)

        one_hot = {f"app_cat_{c}": (1 if c == app_cat else 0) for c in app_cats}

        row = {
            **{f"embed_{i}": 0.0 for i in range(128)},
            **one_hot,
            "sender_open_rate": s_open,
            "sender_avg_response_time_sec": s_resp,
            "sender_interaction_frequency": s_freq,
            "sender_recency_days": s_recency,
            "sender_conversation_ratio": s_conv,
            "app_open_rate": app_open,
            "app_daily_volume": app_volume,
            "hour_of_day": float(hour),
            "is_weekend": float(is_weekend),
            "is_sleep_hours": float(is_sleep),
            "minutes_since_last_from_sender": mins_since,
            "is_group_conversation": float(is_group),
            "notification_has_reply_action": float(has_reply),
            "notification_importance_raw": float(notif_importance_raw),
            "is_messaging_style": float(is_messaging_style),
            "flag_foreground_service": float(flag_fs),
            "title": title,
            "text": body,
            "package_name": pkg,
            "app_category": app_cat,
            "label_importance": imp,
            "label_urgency": urg,
            "label_category": cat,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure all FEATURE_COLUMNS are present
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df


if __name__ == "__main__":
    df = generate_dataset(5000, seed=42)
    df.to_json("data/training_data.json", orient="records", indent=2)
    print(f"Generated {len(df)} samples")
    print(df["label_category"].value_counts())
