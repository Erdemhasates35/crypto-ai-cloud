#!/usr/bin/env python3
"""
ULTRA CLOUD AI TRADING SYSTEM
Erdemhasates35 - GitHub Actions Cloud Power
Real Trading with Advanced AI
"""
import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# AI Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

print("🚀 ULTRA CLOUD AI STARTING...")
print(f"👤 Developer: Erdemhasates35")
print(f"🕐 UTC: {datetime.utcnow()}")
print("☁️  Platform: GitHub Actions")

# API credentials from secrets
BINANCE_API = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET = os.getenv('BINANCE_SECRET_KEY')
BITGET_API = os.getenv('BITGET_API_KEY')
BITGET_SECRET = os.getenv('BITGET_SECRET_KEY')
BITGET_PASS = os.getenv('BITGET_PASSPHRASE')

print(f"🔑 API Keys Status:")
print(f"   Binance API: {'✅ Found' if BINANCE_API else '❌ Missing'}")
print(f"   Bitget API: {'✅ Found' if BITGET_API else '❌ Missing'}")

if not BINANCE_API:
    print("❌ No API keys found. Please set GitHub Secrets.")
    exit(1)

class CloudAITrader:
    def __init__(self):
        self.binance = ccxt.binance({
            'apiKey': BINANCE_API,
            'secret': BINANCE_SECRET,
            'sandbox': False,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'recvWindow': 60000
            }
        })
        
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.initial_capital = 20.0
        self.current_capital = 20.0
        self.trades_executed = 0
        self.successful_trades = 0
        self.session_start = datetime.utcnow()
        
        # AI Models
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.nn_model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
        self.scaler = StandardScaler()
        
    async def test_connection(self):
        """API bağlantısını test et"""
        try:
            balance = await asyncio.to_thread(self.binance.fetch_balance)
            usdt_balance = balance.get('USDT', {}).get('total', 0)
            print(f"✅ Binance connected. USDT Balance: {usdt_balance}")
            self.current_capital = float(usdt_balance) if usdt_balance > 0 else self.initial_capital
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def collect_data(self):
        """Market verisi topla"""
        market_data = {}
        
        for symbol in self.symbols:
            try:
                ohlcv = await asyncio.to_thread(
                    self.binance.fetch_ohlcv, symbol, '5m', limit=200
                )
                
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df = self.add_indicators(df)
                    market_data[symbol] = df
                    print(f"📊 {symbol}: {len(df)} data points")
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"❌ {symbol} data error: {e}")
        
        return market_data
    
    def add_indicators(self, df):
        """Teknik indikatörler"""
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - ema_26
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price features
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(10).std()
        
        return df.dropna()
    
    async def train_models(self, market_data):
        """AI modellerini eğit"""
        X_train = []
        y_train = []
        
        for symbol, df in market_data.items():
            if len(df) > 50:
                features = ['sma_20', 'ema_12', 'rsi', 'macd', 'volume_ratio', 'price_change', 'volatility']
                df['target'] = df['close'].shift(-3).pct_change()
                
                df_clean = df[features + ['target']].dropna()
                
                if len(df_clean) > 20:
                    X_train.extend(df_clean[features].values.tolist())
                    y_train.extend(df_clean['target'].values.tolist())
        
        if len(X_train) > 50:
            X_scaled = self.scaler.fit_transform(X_train)
            
            self.rf_model.fit(X_scaled, y_train)
            self.gb_model.fit(X_scaled, y_train)
            self.nn_model.fit(X_scaled, y_train)
            
            print(f"🧠 Models trained with {len(X_train)} samples")
            return True
        
        return False
    
    async def analyze_symbol(self, symbol, df):
        """Sembol analizi"""
        if len(df) < 10:
            return None
        
        latest_row = df.iloc[-1]
        features = [
            latest_row.get('sma_20', 0),
            latest_row.get('ema_12', 0),
            latest_row.get('rsi', 50),
            latest_row.get('macd', 0),
            latest_row.get('volume_ratio', 1),
            latest_row.get('price_change', 0),
            latest_row.get('volatility', 0.01)
        ]
        
        features_scaled = self.scaler.transform([features])
        
        # AI predictions
        rf_pred = self.rf_model.predict(features_scaled)[0]
        gb_pred = self.gb_model.predict(features_scaled)[0]
        nn_pred = self.nn_model.predict(features_scaled)[0]
        
        # Ensemble
        ensemble_pred = (rf_pred + gb_pred + nn_pred) / 3
        
        # Confidence
        predictions = [rf_pred, gb_pred, nn_pred]
        confidence = 1.0 - np.std(predictions)
        confidence = max(0.1, min(0.95, confidence))
        
        return {
            'symbol': symbol,
            'prediction': ensemble_pred,
            'confidence': confidence,
            'current_price': float(latest_row.get('close', 0))
        }
    
    async def execute_trade(self, analysis):
        """Trade yürüt"""
        try:
            symbol = analysis['symbol']
            prediction = analysis['prediction']
            confidence = analysis['confidence']
            
            if abs(prediction) < 0.003 or confidence < 0.6:
                return {'success': False, 'reason': 'Weak signal'}
            
            # Position sizing
            position_value = self.current_capital * 0.1 * confidence
            leverage = min(10, max(1, int(confidence * 10)))
            
            current_price = analysis['current_price']
            position_size = (position_value * leverage) / current_price
            
            if position_size < 0.001:
                return {'success': False, 'reason': 'Position too small'}
            
            side = 'LONG' if prediction > 0 else 'SHORT'
            
            print(f"🎯 EXECUTING TRADE:")
            print(f"   {symbol}: {side}")
            print(f"   Size: {position_size:.6f}")
            print(f"   Leverage: {leverage}x")
            print(f"   Prediction: {prediction*100:.3f}%")
            print(f"   Confidence: {confidence:.3f}")
            
            # Set leverage
            await asyncio.to_thread(self.binance.set_leverage, leverage, symbol)
            
            # Execute order
            if side == 'LONG':
                order = await asyncio.to_thread(
                    self.binance.create_market_buy_order, symbol, position_size
                )
            else:
                order = await asyncio.to_thread(
                    self.binance.create_market_sell_order, symbol, position_size
                )
            
            self.trades_executed += 1
            print(f"✅ Trade executed! Order ID: {order.get('id')}")
            
            return {'success': True, 'order': order}
            
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
            return {'success': False, 'reason': str(e)}
    
    async def save_results(self):
        """Sonuçları kaydet"""
        runtime_minutes = (datetime.utcnow() - self.session_start).total_seconds() / 60
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'developer': 'Erdemhasates35',
            'platform': 'GitHub Actions',
            'runtime_minutes': runtime_minutes,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return_percent': total_return,
            'trades_executed': self.trades_executed,
            'successful_trades': self.successful_trades,
            'success_rate': (self.successful_trades / max(1, self.trades_executed)) * 100
        }
        
        with open('trading_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        with open('session.log', 'w') as f:
            f.write(f"Cloud AI Trading Session - Erdemhasates35\n")
            f.write(f"Start: {self.session_start}\n")
            f.write(f"End: {datetime.utcnow()}\n")
            f.write(f"Runtime: {runtime_minutes:.1f} minutes\n")
            f.write(f"Capital: ${self.initial_capital} → ${self.current_capital:.2f}\n")
            f.write(f"Return: {total_return:.2f}%\n")
            f.write(f"Trades: {self.trades_executed}\n")
        
        print(f"📊 Results saved. Return: {total_return:.2f}%")
    
    async def run_session(self):
        """Ana trading session"""
        print("🚀 Starting Cloud AI Trading Session...")
        
        # Test connection
        if not await self.test_connection():
            return
        
        # Collect data
        market_data = await self.collect_data()
        
        # Train models
        if not await self.train_models(market_data):
            print("❌ Model training failed")
            return
        
        # Execute trades
        for symbol in self.symbols:
            if symbol in market_data:
                analysis = await self.analyze_symbol(symbol, market_data[symbol])
                
                if analysis:
                    result = await self.execute_trade(analysis)
                    if result['success']:
                        self.successful_trades += 1
                
                await asyncio.sleep(2)
        
        # Update final capital
        await self.test_connection()
        
        # Save results
        await self.save_results()
        
        print(f"🏁 Session completed!")
        print(f"   Capital: ${self.initial_capital} → ${self.current_capital:.2f}")
        print(f"   Trades: {self.trades_executed} (Success: {self.successful_trades})")

if __name__ == "__main__":
    trader = CloudAITrader()
    asyncio.run(trader.run_session())
