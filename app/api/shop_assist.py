import logging
import pandas as pd
import google.generativeai as genai
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException
import re
from app.models.schemas import IntentType, ChatResponse, UserRequest, UserProfile


class LaptopFeatures:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.weights = {
            'gaming': {'gpu': 0.4, 'ram': 0.3, 'processor': 0.3},
            'productivity': {'battery': 0.4, 'weight': 0.3, 'processor': 0.3},
            'student': {'price': 0.4, 'battery': 0.3, 'weight': 0.3},
            'content_creation': {'gpu': 0.3, 'ram': 0.3, 'display': 0.4}
        }


class LaptopShopAssistant:
    def __init__(self, csv_path: str):
        self._setup_logging()
        self._load_data(csv_path)
        self._setup_ai()
        self.features = LaptopFeatures(self.df)
        self.session_data = {}

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_data(self, csv_path: str):
        try:
            self.df = pd.read_csv(csv_path)
            self._preprocess_data()
            self._compute_laptop_vectors()
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise

    def _setup_ai(self):
        api_key = 'AIzaSyDrnUNdl2sQO1wT8nPL_rpHik-pVmR9E6Y'  # os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def _preprocess_data(self):
        column_mapping = {
            'Brand': 'brand',
            'Model Name': 'model_name',
            'Core': 'core',
            'CPU Manufacturer': 'cpu_manufacturer',
            'Clock Speed': 'clock_speed',
            'RAM Size': 'ram_size',
            'Storage Type': 'storage_type',
            'Display Type': 'display_type',
            'Display Size': 'display_size',
            'Graphics Processor': 'graphics_processor',
            'Screen Resolution': 'screen_resolution',
            'OS': 'os',
            'Laptop Weight': 'laptop_weight',
            'Special Features': 'special_features',
            'Average Battery Life': 'battery_life',
            'Price': 'price_inr',
            'Description': 'product_description'
        }

        self.df.rename(columns=column_mapping, inplace=True)

        # Process numerical values
        self.df['ram_gb'] = self.df['ram_size'].str.extract(r'(\d+)').astype(float)
        self.df['weight_kg'] = self.df['laptop_weight'].str.extract(r'(\d+\.?\d*)').astype(float)
        self.df['battery_hours'] = pd.to_numeric(self.df['battery_life'], errors='coerce')
        self.df['gpu_score'] = self.df['graphics_processor'].apply(self._score_gpu)
        self.df['display_inches'] = self.df['display_size'].str.extract(r'(\d+\.?\d*)').astype(float)
        self.df['cpu_ghz'] = self.df['clock_speed'].str.extract(r'(\d+\.?\d*)').astype(float)
        self.df['price_inr'] = pd.to_numeric(self.df['price_inr'].str.replace(',', ''), errors='coerce')

    def _compute_laptop_vectors(self):
        features = ['price_inr', 'ram_gb', 'weight_kg', 'gpu_score', 'battery_hours', 'display_inches', 'cpu_ghz']
        feature_matrix = self.df[features].fillna(0)
        self.scaler = MinMaxScaler()
        self.laptop_vectors = self.scaler.fit_transform(feature_matrix)

    def _score_gpu(self, gpu: str) -> float:
        gpu = str(gpu).lower()
        scores = {
            'rtx': 5.0,
            'nvidia': 4.0,
            'radeon': 3.5,
            'iris': 2.5
        }
        return next((score for key, score in scores.items() if key in gpu), 1.0)
    #IntentType.SPECS: ['specs', 'specifications', 'features', 'details'],
    def _classify_intent(self, message: str) -> Tuple[IntentType, float]:
        message = message.lower()
        patterns = {
            IntentType.PRICING: ['under', 'below', 'price', 'cost', 'between', 'budget'],
            IntentType.PURCHASE: ['gaming', 'graphics', 'game', 'buy', 'recommend', 'suggest'],
            IntentType.COMPARE: ['compare', 'difference', 'versus', 'vs', 'better'],

            IntentType.SPECS: [
                'battery', 'battery life', 'long lasting',
                'battery backup', 'portable', 'duration' , 'specs', 'specifications', 'features', 'details'
            ],
            IntentType.SUPPORT: ['help', 'support', 'issue', 'problem', 'not working'],
            IntentType.TECHNICAL: ['how to', 'technical', 'explain', 'performance']
        }

        for intent, words in patterns.items():
            if any(word in message for word in words):
                return intent, 0.9

        prompt = f"Classify intent as purchase, compare, specs, support, pricing, technical, or unknown: {message}"
        response = self.model.generate_content(prompt)
        return IntentType(response.text.strip().lower()), 0.7

    async def process_request(self, request: UserRequest) -> ChatResponse:
        try:
            intent, confidence = self._classify_intent(request.message)
            self._update_session(request)
            response = await self._handle_intent(intent, request)
            response.confidence_score = confidence
            return response
        except Exception as e:
            self.logger.error(f"Request processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _update_session(self, request: UserRequest):
        if request.session_id not in self.session_data:
            self.session_data[request.session_id] = []
        self.session_data[request.session_id].append(request.message)

    async def _handle_intent(self, intent: IntentType, request: UserRequest) -> ChatResponse:
        if intent == IntentType.SPECS and any(
                word in request.message.lower()
                for word in ['battery', 'battery life', 'long lasting','good battery', 'battery backup', 'portable', 'duration']
        ):
            return await self._handle_battery_search(request)
        handlers = {
            IntentType.PURCHASE: self._handle_purchase,
            IntentType.COMPARE: self._handle_compare,
            IntentType.SPECS: self._handle_specs,
            IntentType.SUPPORT: self._handle_support,
            IntentType.PRICING: self._handle_price_search,
            IntentType.TECHNICAL: self._handle_technical,
            IntentType.UNKNOWN: self._handle_unknown
        }
        return await handlers.get(intent, self._handle_unknown)(request)

    async def _handle_purchase(self, request: UserRequest) -> ChatResponse:
        if "gaming" in request.message.lower():
            filtered_laptops = self.df[self.df['gpu_score'] >= 4.0].sort_values('gpu_score', ascending=False)

            prompt = f"""Recommend gaming laptops from:
            {filtered_laptops[['brand', 'model_name', 'graphics_processor', 'ram_size', 'price_inr']].head().to_dict('records')}"""

            response = self.model.generate_content(prompt)
            return ChatResponse(
                response=response.text,
                recommendations=filtered_laptops.head(5).to_dict('records'),
                intent=IntentType.PURCHASE.value,
                confidence_score=0.9
            )

        user_profile = await self._extract_user_profile(request.session_id)
        if not user_profile:
            return ChatResponse(
                response="Please specify your requirements (GPU, display, portability, budget, primary use).",
                intent=IntentType.PURCHASE.value,
                confidence_score=0.8
            )

        recommendations = self._get_recommendations(user_profile)
        similar_products = self._find_similar_products(recommendations[0] if recommendations else None)
        response_text = await self._generate_recommendation_text(recommendations, user_profile)

        return ChatResponse(
            response=response_text,
            recommendations=recommendations,
            similar_products=similar_products,
            intent=IntentType.PURCHASE.value,
            confidence_score=0.9
        )

    async def _handle_battery_search(self, request: UserRequest) -> ChatResponse:
        try:
            # Filter laptops with good battery life (> 6 hours)
            filtered_laptops = self.df[
                self.df['battery_hours'] >= 3
                ].sort_values('battery_hours', ascending=False)

            if filtered_laptops.empty:
                return ChatResponse(
                    response="No laptops found with extended battery life.",
                    intent=IntentType.SPECS.value,
                    confidence_score=0.9
                )

            # Prepare the analysis
            summary = {
                'total_laptops': len(filtered_laptops),
                'avg_battery': filtered_laptops['battery_hours'].mean(),
                'laptops': filtered_laptops[
                    ['brand', 'model_name', 'battery_life', 'price_inr', 'core', 'ram_size']
                ].head(5).to_dict('records')
            }

            prompt = f"""Analyze these laptops with good battery life:
            Total options: {summary['total_laptops']}
            Average battery life: {summary['avg_battery']:.1f} hours
            Top picks: {json.dumps(summary['laptops'], indent=2)}

            Focus on:
            1. Battery life performance
            2. Portability features
            3. Value for money
            4. Best use cases"""

            response = self.model.generate_content(prompt)
            return ChatResponse(
                response=response.text,
                recommendations=filtered_laptops.head(5).to_dict('records'),
                intent=IntentType.SPECS.value,
                confidence_score=0.9
            )

        except Exception as e:
            self.logger.error(f"Battery search failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Search failed")
    def _get_recommendations(self, profile: UserProfile) -> List[Dict]:
        def score_laptop(laptop):
            score = 0
            # GPU Score
            if profile.gpu_intensity.lower() == 'high':
                score += laptop['gpu_score'] * 2

            # Portability Score
            if profile.portability.lower() == 'high':
                score += (1 / laptop['weight_kg']) if laptop['weight_kg'] > 0 else 0
                score += laptop['battery_hours'] * 0.5 if laptop['battery_hours'] else 0

            # Budget Score
            if laptop['price_inr'] <= profile.budget:
                score += (1 - laptop['price_inr'] / profile.budget) * 3

            # Use Case Scores
            use_case_scores = {
                'gaming': lambda l: l['gpu_score'] * 0.4 + l['ram_gb'] / 32 * 0.3 + l['cpu_ghz'] / 4 * 0.3,
                'productivity': lambda l: l['battery_hours'] / 12 * 0.4 + (1 / l['weight_kg']) * 0.3 + l[
                    'ram_gb'] / 32 * 0.3,
                'student': lambda l: (1 - l['price_inr'] / profile.budget) * 0.4 + l['battery_hours'] / 12 * 0.3 + (
                        1 / l['weight_kg']) * 0.3,
                'content_creation': lambda l: l['gpu_score'] * 0.3 + l['ram_gb'] / 32 * 0.3 + l[
                    'display_inches'] / 17 * 0.4
            }

            if profile.primary_use.lower() in use_case_scores:
                score += use_case_scores[profile.primary_use.lower()](laptop) * 2

            return score

        scored_laptops = [
            {**laptop.to_dict(), 'score': score_laptop(laptop)}
            for _, laptop in self.df.iterrows()
        ]
        return sorted(scored_laptops, key=lambda x: x['score'], reverse=True)[:3]

    def _find_similar_products(self, laptop: Optional[Dict]) -> List[Dict]:
        if not laptop:
            return []

        try:
            laptop_idx = self.df.index[self.df['model_name'] == laptop['model_name']].values[0]
            target_vector = self.laptop_vectors[laptop_idx].reshape(1, -1)
            distances = np.linalg.norm(self.laptop_vectors - target_vector, axis=1)
            similar_indices = distances.argsort()[1:4]

            similar_laptops = []
            for idx in similar_indices:
                laptop_dict = self.df.iloc[idx].to_dict()
                laptop_dict['similarity_score'] = 1 - (distances[idx] / distances.max())
                similar_laptops.append(laptop_dict)

            return similar_laptops
        except Exception as e:
            self.logger.error(f"Similar products search failed: {str(e)}")
            return []

    async def _generate_recommendation_text(self, recommendations: List[Dict], profile: UserProfile) -> str:
        laptops_info = [{
            'brand': r['brand'],
            'model': r['model_name'],
            'price': r['price_inr'],
            'specs': {
                'processor': r['core'],
                'ram': r['ram_size'],
                'gpu': r['graphics_processor'],
                'display': f"{r['display_size']} {r['display_type']}",
                'storage': r['storage_type']
            },
            'score': r.get('score', 0)
        } for r in recommendations]

        prompt = f"""Explain why these laptops match the user profile:
        Profile: {json.dumps(profile.__dict__)}
        Recommendations: {json.dumps(laptops_info)}
        Focus on: performance match, value for money, key advantages"""

        response = self.model.generate_content(prompt)
        return response.text

    async def _extract_user_profile(self, session_id: str) -> Optional[UserProfile]:
        if session_id not in self.session_data:
            return None

        conversation = self.session_data[session_id]
        prompt = f"""Extract user preferences from conversation: {' '.join(conversation)}
       Return strict JSON format:
       {{
           "gpu_intensity": "low/medium/high",
           "display_quality": "standard/premium",
           "portability": "low/high",
           "multitasking": "light/heavy",
           "processing_speed": "basic/advanced",
           "budget": <number>,
           "primary_use": "gaming/productivity/student/content_creation",
           "brand_preference": "<brand_name or null>"
       }}"""

        try:
            response = self.model.generate_content(prompt)
            data = json.loads(response.text)
            return UserProfile(**data)
        except Exception as e:
            self.logger.error(f"Profile extraction failed: {str(e)}")
            return None

    async def _handle_compare(self, request: UserRequest) -> ChatResponse:
        models = self._extract_laptop_models(request.message)
        if len(models) < 2:
            return ChatResponse(
                response="Please specify which laptops to compare.",
                intent=IntentType.COMPARE.value
            )

        compared_laptops = self._get_compared_laptops(models)
        comparison = await self._generate_comparison_text(compared_laptops)
        return ChatResponse(
            response=comparison,
            recommendations=compared_laptops,
            intent=IntentType.COMPARE.value
        )

    def _get_compared_laptops(self, models: List[str]) -> List[Dict]:
        compared_laptops = []
        for model in models:
            laptop = self.df[self.df['model_name'] == model].iloc[0]
            compared_laptops.append({
                'brand': laptop['brand'],
                'model_name': laptop['model_name'],
                'price_inr': float(laptop['price_inr']),
                'core': laptop['core'],
                'ram_size': laptop['ram_gb'],
                'storage_type': laptop['storage_type'],
                'gpu_score': laptop['gpu_score'],
                'display_size': laptop['display_inches'],
                'battery_hours': laptop['battery_hours'],
                'graphics_processor': laptop['graphics_processor'],
                'special_features': laptop['special_features']
            })
        return compared_laptops

    async def _generate_comparison_text(self, laptops: List[Dict]) -> str:
        prompt = f"""Compare laptops:
            {json.dumps(laptops, indent=2)}

            Compare:
            1. Performance (CPU, RAM, GPU)
            2. Display and design
            3. Value for money
            4. Battery life
            5. Best use cases"""

        response = self.model.generate_content(prompt)
        return response.text

    def _extract_laptop_models(self, message: str) -> List[str]:
        models = self.df['model_name'].tolist()
        message = message.lower()

        # Direct model matching
        found_models = [model for model in models if model.lower() in message]

        # Brand + Model matching
        brands = self.df['brand'].unique()
        for brand in brands:
            brand_models = self.df[self.df['brand'] == brand]['model_name'].tolist()
            if brand.lower() in message:
                for model in brand_models:
                    model_parts = model.lower().split()
                    if any(part in message for part in model_parts):
                        found_models.append(model)

        # Clean duplicates
        found_models = list(dict.fromkeys(found_models))

        # Fuzzy matching if no exact matches
        if not found_models:
            words = message.split()
            for model in models:
                model_words = model.lower().split()
                if any(word in words for word in model_words):
                    found_models.append(model)

        return found_models[:2]  # Return max 2 models for comparison```

    async def _handle_specs(self, request: UserRequest) -> ChatResponse:
        models = self._extract_laptop_models(request.message)
        if not models:
            return ChatResponse(
                response="Please specify the laptop model.",
                intent=IntentType.SPECS.value
            )

        laptop = self.df[self.df['model_name'] == models[0]].iloc[0]
        specs = {
            'brand': laptop['brand'],
            'model': laptop['model_name'],
            'processor': {
                'core': laptop['core'],
                'manufacturer': laptop['cpu_manufacturer'],
                'speed': laptop['clock_speed']
            },
            'memory': laptop['ram_size'],
            'storage': laptop['storage_type'],
            'display': {
                'size': laptop['display_size'],
                'type': laptop['display_type'],
                'resolution': laptop['screen_resolution']
            },
            'graphics': laptop['graphics_processor'],
            'battery': laptop['battery_life']
        }

        analysis = await self._generate_specs_analysis(specs)
        return ChatResponse(
            response=analysis,
            recommendations=[laptop.to_dict()],
            intent=IntentType.SPECS.value
        )

    async def _generate_specs_analysis(self, specs: Dict) -> str:
        prompt = f"""Analyze laptop specifications:
       {json.dumps(specs, indent=2)}

       Provide:
       1. Performance analysis (CPU/GPU/RAM)
       2. Display quality assessment
       3. Storage configuration
       4. Battery and portability
       5. Key features and advantages
       6. Best use cases
       7. Value proposition"""

        response = self.model.generate_content(prompt)
        return response.text

    async def _handle_price_search(self, request: UserRequest) -> ChatResponse:
        price_range = self._extract_price_range(request.message)
        filtered_laptops = self.df[
            (self.df['price_inr'] >= price_range['min']) &
            (self.df['price_inr'] <= price_range['max'])
            ].sort_values('price_inr')

        if filtered_laptops.empty:
            return ChatResponse(
                response=f"No laptops found in range ₹{price_range['min']:,} - ₹{price_range['max']:,}",
                intent=IntentType.PRICING.value
            )

        analysis = await self._analyze_price_range(filtered_laptops, price_range)
        return ChatResponse(
            response=analysis,
            recommendations=filtered_laptops.head(5).to_dict('records'),
            intent=IntentType.PRICING.value
        )

    async def _analyze_price_range(self, filtered_df: pd.DataFrame, price_range: Dict[str, float]) -> str:
        summary = {
            'total_laptops': len(filtered_df),
            'price_range': f"₹{price_range['min']:,} - ₹{price_range['max']:,}",
            'avg_price': filtered_df['price_inr'].mean(),
            'segments': {
                'budget': filtered_df[
                    filtered_df['price_inr'] <= price_range['min'] + (price_range['max'] - price_range['min']) / 3],
                'mid': filtered_df[
                    (filtered_df['price_inr'] > price_range['min'] + (price_range['max'] - price_range['min']) / 3) &
                    (filtered_df['price_inr'] <= price_range['min'] + 2 * (
                            price_range['max'] - price_range['min']) / 3)],
                'premium': filtered_df[
                    filtered_df['price_inr'] > price_range['min'] + 2 * (price_range['max'] - price_range['min']) / 3]
            },
            'recommendations': filtered_df[
                ['brand', 'model_name', 'price_inr', 'core', 'ram_size', 'graphics_processor']].head(5).to_dict(
                'records')
        }

        prompt = f"""Analyze laptops in price range {summary['price_range']}:
       Total options: {summary['total_laptops']}
       Average price: ₹{summary['avg_price']:,.2f}
       Recommendations: {json.dumps(summary['recommendations'], indent=2)}

       1. Value for money options
       2. Features at different price points
       3. Best choices for different use cases"""

        response = self.model.generate_content(prompt)
        return response.text

    def _extract_price_range(self, message: str) -> Dict[str, float]:
        message = message.lower()
        numbers = [float(n) for n in re.findall(r'\d+(?:,\d+)*', message.replace(',', ''))]

        price_range = {'min': 0, 'max': float('inf')}

        if not numbers:
            return price_range

        if 'under' in message or 'below' in message or 'less than' in message:
            price_range['max'] = numbers[0]
        elif 'above' in message or 'over' in message or 'more than' in message:
            price_range['min'] = numbers[0]
        elif 'between' in message and len(numbers) >= 2:
            price_range['min'] = min(numbers[0], numbers[1])
            price_range['max'] = max(numbers[0], numbers[1])
        else:
            price_range['max'] = numbers[0]

        return price_range

    async def _handle_support(self, request: UserRequest) -> ChatResponse:
        prompt = f"Provide technical support guidance for: {request.message}"
        guidance = self.model.generate_content(prompt).text
        return ChatResponse(
            response=guidance,
            intent=IntentType.SUPPORT.value
        )

    async def _handle_technical(self, request: UserRequest) -> ChatResponse:
        prompt = f"Explain technical aspects of: {request.message}"
        explanation = self.model.generate_content(prompt).text
        return ChatResponse(
            response=explanation,
            intent=IntentType.TECHNICAL.value
        )

    async def _handle_unknown(self, request: UserRequest) -> ChatResponse:
        return ChatResponse(
            response="Could you please rephrase your question?",
            intent=IntentType.UNKNOWN.value
        )

