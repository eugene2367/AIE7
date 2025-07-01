# Prompt Engineering and Prototyping Best Practices - Activities

## Activity #1: System Evaluation

Please evaluate your system on the following questions:

### Question 1
**Question:** Explain the concept of object-oriented programming in simple terms to a complete beginner.
Answer : Sure! Object-oriented programming (often called OOP) is a way of writing computer programs that makes it easier to organize and manage your code.

Imagine you want to create a video game with characters, cars, and houses. Instead of writing everything from scratch over and over, OOP lets you create "blueprints" called *objects that represent real things.

Here's how it works simply:

1. Objects: Think of an object like a thing or a noun — for example, a dog, a car, or a person. Each object can have:
   - Attributes (or properties): These are details about the object, like a dog's color or size.
   - Methods (or actions): These are things the object can do, like a dog barking or running.

2. Classes: A class is like a blueprint or template for creating objects. For example, you might have a class called "Dog" that defines what a dog is and what it can do. Then you can create many dog objects using that blueprint — each dog can have different attributes (like different colors or names).

3. Why use it? With OOP, you can:
   - Reuse code by creating many objects from one class.
   - Keep your code organized.
   - Make it easier to fix and update because you can work on one object without messing up others.

Example:*  
If you have a "Car" class, it might include:
- Attributes: color, brand, speed  
- Methods: drive(), stop(), honk()

Then you create objects like "myCar" or "yourCar" from the Car class, each with their own color and speed.

That's the basic idea of object-oriented programming — using objects and classes to model and organize your code in a way that's similar to real-world things!
- **Aspect Tested:**  : Pedagogical Explainability / Technical Simplification – can the model teach a technical idea to a novice using clear analogies and correct terminology?
- **System Performance Evaluation:** Straightforward explanation, no issues


### Question 2
**Question:** Read the following paragraph and provide a concise summary of the key points…
**input** https://newsroom.aboutrobinhood.com/robinhood-launches-stock-tokens-reveals-layer-2-blockchain-and-expands-crypto-suite-in-eu-and-us-with-perpetual-futures-and-staking/
**output**1. Stock Tokens in the EU
   - Robinhood has introduced US stock and ETF tokens for eligible customers across 30 EU and EEA countries.
   - These tokens provide exposure to over 200 US stocks and ETFs, with zero commissions or added spreads from Robinhood (some fees may still apply).
   - Features include 24/5 trading access and dividend payments directly to token holders via the app.
   - This marks a transition of the European Robinhood app from crypto-only to an all-in-one investment platform powered by crypto.

2. Robinhood Layer 2 Blockchain
   - Robinhood revealed it is developing its own Layer 2 blockchain based on Arbitrum technology.
   - Initially, stock tokens are issued on Arbitrum; going forward, Robinhood's Layer 2 blockchain will power tokenization of real-world assets.
   - The blockchain will support 24/7 trading, seamless bridging, and self-custody and will be optimized for tokenized assets.

3. Crypto Perpetual Futures in the EU
   - Robinhood is launching crypto perpetual futures contracts for eligible EU customers.
   - Offers continuous exposure with up to 3x leverage.
   - The trading interface is designed to simplify complexity with intuitive controls for position and margin management.
   - Orders will be routed via Bitstamp's perpetual futures exchange.
   - Full rollout to eligible customers expected by end of summer 2025.

4. Crypto Staking in the US
   - Staking is now available to eligible US customers starting with Ethereum and Solana.
   - Also available across EU and EEA customers.
   - Enables participation in blockchain networks to earn competitive rewards with an easy-to-use interface.

5. Expanded Crypto and Trading Features
   - Instant Boost on Crypto Deposits: For a limited time, US and EU investors depositing crypto will earn a 1% bonus, potentially up to 2% if $500 million in deposits is reached platform-wide.
   - Robinhood Gold Credit Card: US customers receive cash back on purchases, with plans to enable automatic crypto purchase with rewards in fall 2025.
   - Cortex AI Assistant: Launching later this year in the US for Robinhood Gold members, offering AI-powered insights, market trends, and analysis on tokens.
   - Smart Exchange Routing: Routes orders across partner exchanges to secure best prices and will introduce fee tier discounts based on trading volume. API access coming soon.
   - Tax Lots Feature: US customers can now view and sell specific tax lots for crypto trades for tax strategy benefits.
   - Advanced Charts: Robinhood Legend advanced charts will come to mobile first for equities, expanding to crypto by August 2025.

Regulatory and Legal Notes
- Robinhood Europe, UAB is authorized and regulated by the Bank of Lithuania.
- In the US, Robinhood Crypto, LLC operates under a virtual currency license from New York State DFS.
- Stock tokens, crypto futures, and staking involve risks and are subject to eligibility and restrictions.
- Crypto held on Robinhood is not FDIC insured or SIPC protected.

About Robinhood
- Robinhood Markets, Inc. (NASDAQ: HOOD) pioneered commission-free trading and offers stocks, options, futures, and crypto products.
- Headquartered in Menlo Park, California, Robinhood aims to make investing accessible, modern, and technology-driven.

Forward Looking Statements*
- The announcement contains forward-looking statements subject to risks, including regulatory changes and market conditions.
- Actual results may differ materially from expectations.

---

For detailed information, product eligibility, fees, and terms, visit: [go.robinhood.com/presents](https://go.robinhood.com/presents)
- **Aspect Tested:** : Reading Comprehension & Abstractive Summarization – can it distill key points while preserving nuance?	
- **System Performance Evaluation:** Did a relatively good job of summarizing the TLDR of the new article

### Question 3
**Question:** Write a short, imaginative story (100–150 words) about a robot finding friendship in an unexpected place.
**output** In a forgotten scrapyard on the city's edge, a rusty robot named Jax stirred to life. Unlike the sleek models in the bustling metropolis, Jax was clunky and old, long abandoned by his creators. Each day, he scavenged for spare parts, dreaming of purpose.

One rainy evening, Jax discovered a small, injured bird trapped beneath twisted metal. Gently, he lifted the fragile creature with his worn hands and fashioned a tiny splint from scraps. Days passed, and the bird—whom Jax named Pip—grew stronger, returning each morning with a song that echoed through the silent junkyard.

Their unlikely friendship blossomed: Jax provided shelter and protection, while Pip offered warmth and the joy of companionship. In a world that had forgotten him, Jax found a heart beating alongside his mechanical one, proving that even in the most unexpected places, friendship could spark life anew.
- **Aspect Tested:** : Creative Generation & Constraint-following – narrative imagination under a hard word-count limit.
- **System Performance Evaluation:** 142 words, and not too bad, get the job done

### Question 4
**Question:** If a store sells apples in packs of 4 and oranges in packs of 3, how many packs of each do I need to buy to get exactly 12 apples and 9 oranges?
**output** To get exactly 12 apples, since each pack has 4 apples:
- Number of apple packs = 12 ÷ 4 = 3 packs

To get exactly 9 oranges, since each pack has 3 oranges:
- Number of orange packs = 9 ÷ 3 = 3 packs

*Answer:  
You need to buy 3 packs of apples and 3 packs of oranges*.
- **Aspect Tested:** Basic Arithmetic & Logical Reasoning – translates a word problem into simple algebra.	
- **System Performance Evaluation:** Correctly answers the questions and provided the logic

### Question 5
**Question:** Rewrite the following paragraph in a professional, formal tone…
**input**Team,

I'm thrilled to share that after months of cross-functional collaboration and late-night refinements, we successfully pushed NovaPay 2.0 live this morning—on schedule and within budget. Early telemetry shows a 23 % faster checkout flow and zero critical errors in the first six hours, validating the QA team's hard work and everyone's attention to detail. Over the next 48 hours, Product and Support will monitor user feedback channels and core metrics while Engineering remains on "green-button" standby for hotfixes. Marketing's announcement campaign goes live at 10 a.m. EST tomorrow, so please keep Slack notifications on for any real-time adjustments. Thank you for the relentless focus and collaboration; this launch sets a new bar for execution excellence. Let's celebrate the win and stay alert to ensure a seamless rollout for our customers.
**output** I am pleased to announce that, following months of cross-functional collaboration and meticulous refinements, NovaPay 2.0 was successfully launched this morning—on schedule and within budget. Initial telemetry indicates a 23% improvement in checkout speed and no critical errors detected during the first six hours, affirming the quality assurance team's diligent efforts and the collective attention to detail.

Over the next 48 hours, the Product and Support teams will closely monitor user feedback channels and key performance metrics, while the Engineering team remains on standby to implement any necessary hotfixes. Additionally, the Marketing team's announcement campaign is scheduled to commence at 10 a.m. EST tomorrow. Please ensure that Slack notifications remain enabled to facilitate any real-time adjustments.

Thank you all for your unwavering dedication and collaboration. This successful launch sets a new standard for execution excellence. Let us take a moment to acknowledge this achievement while remaining vigilant to guarantee a seamless experience for our customers.
- **Aspect Tested:** Style Transfer & Paraphrase Fidelity – shift register without losing meaning.	
- **System Performance Evaluation:** Did a decent job. however, i believe more data is needed in order for true tone and style to be reflected. seems hard to 0 shot this.

## Discussion Question #1
What are some limitations of vibe checking as an evaluation tool?

It is qualtiative and not objective, no baseline to measure against. Need to implement a quantitative approach.
