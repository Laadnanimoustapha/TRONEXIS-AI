// Enhanced Supply Chain Simulation System - TypeScript Only

// ============================================================================
// INTERFACES AND TYPES
// ============================================================================

interface Product {
  id: string;
  name: string;
  productionCost: number;
  sellingPrice: number;
  weight: number; // kg per unit
}

interface ProductionStats {
  produced: number;
  cost: number;
  defective: number;
}

interface TransferStats {
  sent: number;
  received: number;
  cost: number;
}

interface SalesStats {
  sold: number;
  revenue: number;
  stockouts: number;
}

interface MaintenanceEvent {
  day: number;
  entityId: string;
  entityName: string;
  cost: number;
  downtime: number;
}

interface MarketEvent {
  day: number;
  type: 'demand_surge' | 'demand_drop' | 'price_increase' | 'price_decrease' | 'supply_disruption';
  description: string;
  affectedEntity: string;
  impact: number;
}

// ============================================================================
// FACTORY CLASS (Enhanced)
// ============================================================================

class Factory {
  id: string;
  name: string;
  productionRate: number;
  currentStock: number;
  productionCostPerUnit: number;
  qualityRate: number; // 0.0 to 1.0 (percentage of non-defective products)
  maintenanceCycle: number; // days between maintenance
  private totalProduced: number = 0;
  private totalProductionCost: number = 0;
  private totalDefective: number = 0;
  private daysSinceMaintenanceCheck: number = 0;
  private totalMaintenanceCost: number = 0;
  private isOperational: boolean = true;
  private downtime: number = 0;

  constructor(
    id: string,
    name: string,
    productionRate: number,
    productionCostPerUnit: number,
    qualityRate: number = 0.95,
    maintenanceCycle: number = 7
  ) {
    this.id = id;
    this.name = name;
    this.productionRate = productionRate;
    this.currentStock = 0;
    this.productionCostPerUnit = productionCostPerUnit;
    this.qualityRate = qualityRate;
    this.maintenanceCycle = maintenanceCycle;
  }

  produce(): ProductionStats {
    if (!this.isOperational) {
      this.downtime--;
      if (this.downtime <= 0) {
        this.isOperational = true;
      }
      return { produced: 0, cost: 0, defective: 0 };
    }

    const produced = this.productionRate;
    const goodUnits = Math.floor(produced * this.qualityRate);
    const defective = produced - goodUnits;
    const cost = produced * this.productionCostPerUnit;
    
    this.currentStock += goodUnits;
    this.totalProduced += goodUnits;
    this.totalDefective += defective;
    this.totalProductionCost += cost;
    this.daysSinceMaintenanceCheck++;

    return { produced: goodUnits, cost, defective };
  }

  performMaintenance(): MaintenanceEvent | null {
    if (this.daysSinceMaintenanceCheck >= this.maintenanceCycle) {
      const maintenanceCost = this.productionRate * 2; // Cost scales with production rate
      this.totalMaintenanceCost += maintenanceCost;
      this.daysSinceMaintenanceCheck = 0;
      this.qualityRate = Math.min(0.98, this.qualityRate + 0.02); // Improve quality
      
      return {
        day: 0, // Will be set by simulator
        entityId: this.id,
        entityName: this.name,
        cost: maintenanceCost,
        downtime: 0
      };
    }
    return null;
  }

  setDowntime(days: number): void {
    this.isOperational = false;
    this.downtime = days;
  }

  transferOut(amount: number): number {
    const transferred = Math.min(amount, this.currentStock);
    this.currentStock -= transferred;
    return transferred;
  }

  setProductionRate(rate: number): void {
    this.productionRate = Math.max(0, rate);
  }

  getStats() {
    return {
      id: this.id,
      name: this.name,
      currentStock: this.currentStock,
      productionRate: this.productionRate,
      totalProduced: this.totalProduced,
      totalProductionCost: this.totalProductionCost,
      totalDefective: this.totalDefective,
      qualityRate: (this.qualityRate * 100).toFixed(1),
      totalMaintenanceCost: this.totalMaintenanceCost,
      isOperational: this.isOperational,
      downtime: this.downtime
    };
  }
}

// ============================================================================
// WAREHOUSE CLASS (Enhanced)
// ============================================================================

class Warehouse {
  id: string;
  name: string;
  capacity: number;
  currentStock: number;
  storageCostPerUnit: number;
  temperature: 'ambient' | 'refrigerated' | 'frozen';
  securityLevel: 'basic' | 'medium' | 'high';
  private totalReceived: number = 0;
  private totalSent: number = 0;
  private totalStorageCost: number = 0;
  private spoilageRate: number = 0.001; // 0.1% daily spoilage
  private totalSpoiled: number = 0;
  private insuranceCost: number = 10; // daily insurance cost

  constructor(
    id: string,
    name: string,
    capacity: number,
    storageCostPerUnit: number,
    temperature: 'ambient' | 'refrigerated' | 'frozen' = 'ambient',
    securityLevel: 'basic' | 'medium' | 'high' = 'medium'
  ) {
    this.id = id;
    this.name = name;
    this.capacity = capacity;
    this.currentStock = 0;
    this.storageCostPerUnit = storageCostPerUnit;
    this.temperature = temperature;
    this.securityLevel = securityLevel;

    // Adjust costs based on type
    if (temperature === 'refrigerated') {
      this.storageCostPerUnit *= 1.5;
      this.insuranceCost *= 1.3;
    } else if (temperature === 'frozen') {
      this.storageCostPerUnit *= 2;
      this.insuranceCost *= 1.5;
    }

    if (securityLevel === 'high') {
      this.insuranceCost *= 1.5;
    }
  }

  receiveGoods(amount: number): number {
    const availableSpace = this.capacity - this.currentStock;
    const received = Math.min(amount, availableSpace);
    
    this.currentStock += received;
    this.totalReceived += received;
    
    return received;
  }

  transferOut(amount: number): number {
    const transferred = Math.min(amount, this.currentStock);
    this.currentStock -= transferred;
    this.totalSent += transferred;
    return transferred;
  }

  calculateStorageCost(): number {
    const storageCost = this.currentStock * this.storageCostPerUnit;
    const totalCost = storageCost + this.insuranceCost;
    this.totalStorageCost += totalCost;
    
    // Calculate spoilage
    const spoiled = Math.floor(this.currentStock * this.spoilageRate);
    this.currentStock -= spoiled;
    this.totalSpoiled += spoiled;
    
    return totalCost;
  }

  setCapacity(capacity: number): void {
    this.capacity = Math.max(0, capacity);
  }

  getStats() {
    return {
      id: this.id,
      name: this.name,
      currentStock: this.currentStock,
      capacity: this.capacity,
      utilization: ((this.currentStock / this.capacity) * 100).toFixed(1),
      totalReceived: this.totalReceived,
      totalSent: this.totalSent,
      totalStorageCost: this.totalStorageCost,
      totalSpoiled: this.totalSpoiled,
      temperature: this.temperature,
      securityLevel: this.securityLevel
    };
  }
}

// ============================================================================
// SHOP CLASS (Enhanced)
// ============================================================================

class Shop {
  id: string;
  name: string;
  currentStock: number;
  demand: number;
  sellingPrice: number;
  marketingBudget: number = 0;
  customerSatisfaction: number = 100; // 0-100 scale
  private totalSold: number = 0;
  private totalRevenue: number = 0;
  private stockoutDays: number = 0;
  private totalMarketingSpent: number = 0;
  private returnRate: number = 0.02; // 2% return rate
  private totalReturns: number = 0;

  constructor(
    id: string,
    name: string,
    demand: number,
    sellingPrice: number
  ) {
    this.id = id;
    this.name = name;
    this.currentStock = 0;
    this.demand = demand;
    this.sellingPrice = sellingPrice;
  }

  receiveGoods(amount: number): number {
    this.currentStock += amount;
    return amount;
  }

  sell(): SalesStats {
    const sellAmount = Math.min(this.demand, this.currentStock);
    const stockout = this.demand - sellAmount;
    
    // Apply marketing effect (increases demand fulfillment)
    const marketingBoost = Math.floor(this.marketingBudget / 10);
    const boostedSellAmount = Math.min(sellAmount + marketingBoost, this.currentStock);
    
    // Calculate returns
    const returns = Math.floor(boostedSellAmount * this.returnRate);
    const netSold = boostedSellAmount - returns;
    
    this.currentStock -= boostedSellAmount;
    this.currentStock += returns; // Returns go back to stock
    this.totalSold += netSold;
    this.totalReturns += returns;
    
    const revenue = netSold * this.sellingPrice;
    this.totalRevenue += revenue;
    
    // Update customer satisfaction
    if (stockout > 0) {
      this.stockoutDays += 1;
      this.customerSatisfaction = Math.max(0, this.customerSatisfaction - 5);
    } else {
      this.customerSatisfaction = Math.min(100, this.customerSatisfaction + 2);
    }

    // Spend marketing budget
    this.totalMarketingSpent += this.marketingBudget;

    return {
      sold: netSold,
      revenue: revenue,
      stockouts: stockout
    };
  }

  setDemand(demand: number): void {
    this.demand = Math.max(0, demand);
  }

  setSellingPrice(price: number): void {
    this.sellingPrice = Math.max(0, price);
  }

  setMarketingBudget(budget: number): void {
    this.marketingBudget = Math.max(0, budget);
  }

  getStats() {
    return {
      id: this.id,
      name: this.name,
      currentStock: this.currentStock,
      demand: this.demand,
      sellingPrice: this.sellingPrice,
      totalSold: this.totalSold,
      totalRevenue: this.totalRevenue,
      stockoutDays: this.stockoutDays,
      customerSatisfaction: this.customerSatisfaction.toFixed(1),
      totalMarketingSpent: this.totalMarketingSpent,
      totalReturns: this.totalReturns,
      returnRate: (this.returnRate * 100).toFixed(1)
    };
  }
}

// ============================================================================
// SUPPLIER CLASS (NEW)
// ============================================================================

class Supplier {
  id: string;
  name: string;
  reliability: number; // 0.0 to 1.0
  rawMaterialCost: number;
  leadTime: number; // days
  private totalSupplied: number = 0;
  private totalCost: number = 0;
  private deliveryFailures: number = 0;

  constructor(
    id: string,
    name: string,
    reliability: number,
    rawMaterialCost: number,
    leadTime: number
  ) {
    this.id = id;
    this.name = name;
    this.reliability = reliability;
    this.rawMaterialCost = rawMaterialCost;
    this.leadTime = leadTime;
  }

  supply(amount: number): { supplied: number; cost: number; success: boolean } {
    const success = Math.random() < this.reliability;
    
    if (!success) {
      this.deliveryFailures++;
      return { supplied: 0, cost: 0, success: false };
    }

    const cost = amount * this.rawMaterialCost;
    this.totalSupplied += amount;
    this.totalCost += cost;

    return { supplied: amount, cost, success: true };
  }

  getStats() {
    return {
      id: this.id,
      name: this.name,
      reliability: (this.reliability * 100).toFixed(1),
      rawMaterialCost: this.rawMaterialCost,
      leadTime: this.leadTime,
      totalSupplied: this.totalSupplied,
      totalCost: this.totalCost,
      deliveryFailures: this.deliveryFailures
    };
  }
}

// ============================================================================
// SUPPLY CHAIN SIMULATOR (Enhanced)
// ============================================================================

class SupplyChainSimulator {
  private factories: Map<string, Factory> = new Map();
  private warehouses: Map<string, Warehouse> = new Map();
  private shops: Map<string, Shop> = new Map();
  private suppliers: Map<string, Supplier> = new Map();
  private transportCostPerUnit: number = 0.5;
  private currentDay: number = 0;
  private dailyReports: string[] = [];
  private maintenanceEvents: MaintenanceEvent[] = [];
  private marketEvents: MarketEvent[] = [];
  private enableRandomEvents: boolean = true;
  private totalTransportCost: number = 0;

  addFactory(factory: Factory): void {
    this.factories.set(factory.id, factory);
  }

  addWarehouse(warehouse: Warehouse): void {
    this.warehouses.set(warehouse.id, warehouse);
  }

  addShop(shop: Shop): void {
    this.shops.set(shop.id, shop);
  }

  addSupplier(supplier: Supplier): void {
    this.suppliers.set(supplier.id, supplier);
  }

  setTransportCost(cost: number): void {
    this.transportCostPerUnit = Math.max(0, cost);
  }

  setRandomEvents(enabled: boolean): void {
    this.enableRandomEvents = enabled;
  }

  private generateRandomEvent(): MarketEvent | null {
    if (!this.enableRandomEvents || Math.random() > 0.15) return null;

    const eventTypes: MarketEvent['type'][] = [
      'demand_surge', 'demand_drop', 'price_increase', 'price_decrease', 'supply_disruption'
    ];
    const type = eventTypes[Math.floor(Math.random() * eventTypes.length)];

    const shops = Array.from(this.shops.values());
    const factories = Array.from(this.factories.values());

    let event: MarketEvent | null = null;

    switch (type) {
      case 'demand_surge':
        if (shops.length > 0) {
          const shop = shops[Math.floor(Math.random() * shops.length)];
          const increase = Math.floor(shop.demand * 0.3);
          shop.setDemand(shop.demand + increase);
          event = {
            day: this.currentDay,
            type,
            description: `Market surge at ${shop.name}! Demand increased by ${increase} units.`,
            affectedEntity: shop.id,
            impact: increase
          };
        }
        break;
      case 'demand_drop':
        if (shops.length > 0) {
          const shop = shops[Math.floor(Math.random() * shops.length)];
          const decrease = Math.floor(shop.demand * 0.2);
          shop.setDemand(Math.max(0, shop.demand - decrease));
          event = {
            day: this.currentDay,
            type,
            description: `Economic downturn at ${shop.name}. Demand decreased by ${decrease} units.`,
            affectedEntity: shop.id,
            impact: -decrease
          };
        }
        break;
      case 'supply_disruption':
        if (factories.length > 0) {
          const factory = factories[Math.floor(Math.random() * factories.length)];
          factory.setDowntime(2);
          event = {
            day: this.currentDay,
            type,
            description: `Equipment failure at ${factory.name}! Down for 2 days.`,
            affectedEntity: factory.id,
            impact: -2
          };
        }
        break;
    }

    if (event) {
      this.marketEvents.push(event);
    }

    return event;
  }

  simulateDay(): string {
    this.currentDay++;
    let report = `\n${"=".repeat(80)}\n`;
    report += `DAY ${this.currentDay} SIMULATION\n`;
    report += `${"=".repeat(80)}\n\n`;

    // Check for random events
    const randomEvent = this.generateRandomEvent();
    if (randomEvent) {
      report += `🎲 RANDOM EVENT: ${randomEvent.description}\n\n`;
    }

    let totalProductionCost = 0;
    let totalStorageCost = 0;
    let totalTransportCost = 0;
    let totalRevenue = 0;
    let totalMaintenanceCost = 0;
    let totalMarketingCost = 0;

    // Maintenance Phase
    report += `--- MAINTENANCE PHASE ---\n`;
    let maintenancePerformed = false;
    this.factories.forEach(factory => {
      const maintenance = factory.performMaintenance();
      if (maintenance) {
        maintenance.day = this.currentDay;
        this.maintenanceEvents.push(maintenance);
        totalMaintenanceCost += maintenance.cost;
        report += `🔧 ${factory.name}: Scheduled maintenance completed (Cost: $${maintenance.cost.toFixed(2)})\n`;
        maintenancePerformed = true;
      }
    });
    if (!maintenancePerformed) {
      report += `No maintenance required today.\n`;
    }

    // Production Phase
    report += `\n--- PRODUCTION PHASE ---\n`;
    this.factories.forEach(factory => {
      const stats = factory.produce();
      totalProductionCost += stats.cost;
      if (stats.produced === 0) {
        report += `⚠️  ${factory.name}: NOT OPERATIONAL (Downtime remaining)\n`;
      } else {
        report += `${factory.name}: Produced ${stats.produced} units`;
        if (stats.defective > 0) {
          report += ` (${stats.defective} defective)`;
        }
        report += ` (Cost: $${stats.cost.toFixed(2)})\n`;
        report += `  Current Stock: ${factory.currentStock} units\n`;
      }
    });

    // Logistics Phase (Factory → Warehouse)
    report += `\n--- LOGISTICS PHASE (Factory → Warehouse) ---\n`;
    this.factories.forEach(factory => {
      const factoryStock = factory.currentStock;
      if (factoryStock > 0) {
        const warehouseArray = Array.from(this.warehouses.values());
        const amountPerWarehouse = Math.floor(factoryStock / warehouseArray.length);
        
        warehouseArray.forEach(warehouse => {
          const transferred = factory.transferOut(amountPerWarehouse);
          const received = warehouse.receiveGoods(transferred);
          const transportCost = transferred * this.transportCostPerUnit;
          totalTransportCost += transportCost;
          this.totalTransportCost += transportCost;
          
          if (received < transferred) {
            report += `${factory.name} → ${warehouse.name}: Transferred ${transferred}, accepted ${received} (capacity limit). Cost: $${transportCost.toFixed(2)}\n`;
          } else {
            report += `${factory.name} → ${warehouse.name}: Transferred ${transferred} units. Cost: $${transportCost.toFixed(2)}\n`;
          }
        });
      }
    });

    // Storage Phase
    report += `\n--- STORAGE PHASE ---\n`;
    this.warehouses.forEach(warehouse => {
      const storageCost = warehouse.calculateStorageCost();
      totalStorageCost += storageCost;
      const stats = warehouse.getStats();
      report += `${warehouse.name}: Storing ${warehouse.currentStock}/${warehouse.capacity} units (${stats.utilization}% full)\n`;
      report += `  Cost: $${storageCost.toFixed(2)} | Type: ${stats.temperature} | Security: ${stats.securityLevel}\n`;
    });

    // Logistics Phase (Warehouse → Shop)
    report += `\n--- LOGISTICS PHASE (Warehouse → Shop) ---\n`;
    this.shops.forEach(shop => {
      const demandToFulfill = shop.demand;
      let remainingDemand = demandToFulfill;
      
      this.warehouses.forEach(warehouse => {
        if (remainingDemand > 0) {
          const transferred = warehouse.transferOut(remainingDemand);
          shop.receiveGoods(transferred);
          const transportCost = transferred * this.transportCostPerUnit;
          totalTransportCost += transportCost;
          this.totalTransportCost += transportCost;
          remainingDemand -= transferred;
          
          if (transferred > 0) {
            report += `${warehouse.name} → ${shop.name}: Transferred ${transferred} units. Cost: $${transportCost.toFixed(2)}\n`;
          }
        }
      });
      
      if (remainingDemand > 0) {
        report += `⚠️  ${shop.name}: Could not fulfill ${remainingDemand} units (Stockout)\n`;
      }
    });

    // Sales Phase
    report += `\n--- SALES PHASE ---\n`;
    this.shops.forEach(shop => {
      const salesStats = shop.sell();
      totalRevenue += salesStats.revenue;
      totalMarketingCost += shop.marketingBudget;
      
      const shopStats = shop.getStats();
      report += `${shop.name}: Sold ${salesStats.sold} units (Revenue: $${salesStats.revenue.toFixed(2)})\n`;
      report += `  Customer Satisfaction: ${shopStats.customerSatisfaction}% | Stock: ${shop.currentStock} units\n`;
      if (salesStats.stockouts > 0) {
        report += `  ⚠️  Stockout: ${salesStats.stockouts} units unmet\n`;
      }
    });

    // Financial Summary
    const totalCost = totalProductionCost + totalStorageCost + totalTransportCost + totalMaintenanceCost + totalMarketingCost;
    const profit = totalRevenue - totalCost;

    report += `\n--- FINANCIAL SUMMARY (Day ${this.currentDay}) ---\n`;
    report += `Production Cost:    $${totalProductionCost.toFixed(2)}\n`;
    report += `Storage Cost:       $${totalStorageCost.toFixed(2)}\n`;
    report += `Transport Cost:     $${totalTransportCost.toFixed(2)}\n`;
    report += `Maintenance Cost:   $${totalMaintenanceCost.toFixed(2)}\n`;
    report += `Marketing Cost:     $${totalMarketingCost.toFixed(2)}\n`;
    report += `Total Cost:         $${totalCost.toFixed(2)}\n`;
    report += `Total Revenue:      $${totalRevenue.toFixed(2)}\n`;
    report += `Net Profit/Loss:    $${profit.toFixed(2)} ${profit >= 0 ? '✓' : '✗'}\n`;

    this.dailyReports.push(report);
    return report;
  }

  simulateMultipleDays(days: number): string {
    let fullReport = "";
    for (let i = 0; i < days; i++) {
      fullReport += this.simulateDay();
    }
    return fullReport;
  }

  getComprehensiveStats(): string {
    let report = `\n${"=".repeat(80)}\n`;
    report += `COMPREHENSIVE SUPPLY CHAIN STATISTICS (After Day ${this.currentDay})\n`;
    report += `${"=".repeat(80)}\n\n`;

    report += `--- FACTORIES ---\n`;
    this.factories.forEach(factory => {
      const stats = factory.getStats();
      report += `${stats.name}:\n`;
      report += `  Production Rate: ${stats.productionRate} units/day | Quality: ${stats.qualityRate}%\n`;
      report += `  Current Stock: ${stats.currentStock} units | Status: ${stats.isOperational ? 'Operational' : `Down (${stats.downtime} days)`}\n`;
      report += `  Total Produced: ${stats.totalProduced} units | Defective: ${stats.totalDefective} units\n`;
      report += `  Production Cost: $${stats.totalProductionCost.toFixed(2)} | Maintenance: $${stats.totalMaintenanceCost.toFixed(2)}\n\n`;
    });

    report += `--- WAREHOUSES ---\n`;
    this.warehouses.forEach(warehouse => {
      const stats = warehouse.getStats();
      report += `${stats.name} [${stats.temperature}, ${stats.securityLevel} security]:\n`;
      report += `  Capacity: ${stats.capacity} units | Utilization: ${stats.utilization}%\n`;
      report += `  Current Stock: ${stats.currentStock} units\n`;
      report += `  Total Received: ${stats.totalReceived} | Sent: ${stats.totalSent} | Spoiled: ${stats.totalSpoiled}\n`;
      report += `  Total Storage Cost: $${stats.totalStorageCost.toFixed(2)}\n\n`;
    });

    report += `--- SHOPS ---\n`;
    this.shops.forEach(shop => {
      const stats = shop.getStats();
      report += `${stats.name}:\n`;
      report += `  Daily Demand: ${stats.demand} units | Price: $${stats.sellingPrice}/unit\n`;
      report += `  Current Stock: ${stats.currentStock} units | Customer Satisfaction: ${stats.customerSatisfaction}%\n`;
      report += `  Total Sold: ${stats.totalSold} units | Returns: ${stats.totalReturns} (${stats.returnRate}%)\n`;
      report += `  Revenue: $${stats.totalRevenue.toFixed(2)} | Marketing Spent: $${stats.totalMarketingSpent.toFixed(2)}\n`;
      report += `  Stockout Days: ${stats.stockoutDays}\n\n`;
    });

    if (this.suppliers.size > 0) {
      report += `--- SUPPLIERS ---\n`;
      this.suppliers.forEach(supplier => {
        const stats = supplier.getStats();
        report += `${stats.name}:\n`;
        report += `  Reliability: ${stats.reliability}% | Lead Time: ${stats.leadTime} days\n`;
        report += `  Material Cost: $${stats.rawMaterialCost}/unit\n`;
        report += `  Total Supplied: ${stats.totalSupplied} units | Cost: $${stats.totalCost.toFixed(2)}\n`;
        report += `  Delivery Failures: ${stats.deliveryFailures}\n\n`;
      });
    }

    if (this.marketEvents.length > 0) {
      report += `--- MARKET EVENTS ---\n`;
      this.marketEvents.forEach(event => {
        report += `Day ${event.day}: ${event.description}\n`;
      });
      report += `\n`;
    }

    return report;
  }

  getOverallSummary(): string {
    let totalProduction = 0;
    let totalProductionCost = 0;
    let totalDefective = 0;
    let totalMaintenanceCost = 0;
    this.factories.forEach(factory => {
      const stats = factory.getStats();
      totalProduction += stats.totalProduced;
      totalProductionCost += stats.totalProductionCost;
      totalDefective += stats.totalDefective;
      totalMaintenanceCost += stats.totalMaintenanceCost;
    });

    let totalStorageCost = 0;
    let totalSpoiled = 0;
    this.warehouses.forEach(warehouse => {
      const stats = warehouse.getStats();
      totalStorageCost += stats.totalStorageCost;
      totalSpoiled += stats.totalSpoiled;
    });

    let totalSold = 0;
    let totalRevenue = 0;
    let totalStockoutDays = 0;
    let totalMarketingSpent = 0;
    let totalReturns = 0;
    let avgCustomerSatisfaction = 0;
    this.shops.forEach(shop => {
      const stats = shop.getStats();
      totalSold += stats.totalSold;
      totalRevenue += stats.totalRevenue;
      totalStockoutDays += stats.stockoutDays;
      totalMarketingSpent += stats.totalMarketingSpent;
      totalReturns += stats.totalReturns;
      avgCustomerSatisfaction += parseFloat(stats.customerSatisfaction);
    });
    avgCustomerSatisfaction /= this.shops.size || 1;

    const totalCost = totalProductionCost + totalStorageCost + this.totalTransportCost + totalMaintenanceCost + totalMarketingSpent;
    const netProfit = totalRevenue - totalCost;
    const profitMargin = totalRevenue > 0 ? ((netProfit / totalRevenue) * 100).toFixed(1) : '0.0';

    let summary = `\n${"=".repeat(80)}\n`;
    summary += `OVERALL SUPPLY CHAIN SUMMARY\n`;
    summary += `${"=".repeat(80)}\n`;
    summary += `Simulation Period: ${this.currentDay} days\n`;
    summary += `Random Events: ${this.enableRandomEvents ? 'Enabled' : 'Disabled'} (${this.marketEvents.length} events occurred)\n\n`;
    summary += `--- OPERATIONS ---\n`;
    summary += `Total Produced:     ${totalProduction} units\n`;
    summary += `Total Sold:         ${totalSold} units\n`;
    summary += `Defective Units:    ${totalDefective} units\n`;
    summary += `Spoiled Units:      ${totalSpoiled} units\n`;
    summary += `Returned Units:     ${totalReturns} units\n`;
    summary += `Stockout Days:      ${totalStockoutDays}\n\n`;
    summary += `--- FINANCIAL SUMMARY ---\n`;
    summary += `Total Revenue:      $${totalRevenue.toFixed(2)}\n`;
    summary += `Total Costs:        $${totalCost.toFixed(2)}\n`;
    summary += `  Production:       $${totalProductionCost.toFixed(2)}\n`;
    summary += `  Storage:          $${totalStorageCost.toFixed(2)}\n`;
    summary += `  Transport:        $${this.totalTransportCost.toFixed(2)}\n`;
    summary += `  Maintenance:      $${totalMaintenanceCost.toFixed(2)}\n`;
    summary += `  Marketing:        $${totalMarketingSpent.toFixed(2)}\n`;
    summary += `Net Profit:         $${netProfit.toFixed(2)}\n`;
    summary += `Profit Margin:      ${profitMargin}%\n\n`;
    summary += `--- PERFORMANCE METRICS ---\n`;
    summary += `Avg Customer Satisfaction: ${avgCustomerSatisfaction.toFixed(1)}%\n`;
    summary += `Supply Chain Efficiency:   ${((totalSold / (totalProduction - totalDefective - totalSpoiled)) * 100).toFixed(1)}%\n`;
    summary += `Quality Rate:              ${(((totalProduction - totalDefective) / totalProduction) * 100).toFixed(1)}%\n`;
    summary += `Return Rate:               ${((totalReturns / totalSold) * 100).toFixed(1)}%\n`;

    return summary;
  }

  // ============================================================================
  // DEMONSTRATION AND TESTING
  // ============================================================================

  static runDemo(): void {
    console.log("🚀 SUPPLY CHAIN SIMULATION DEMO\n");

    // Create simulator instance
    const simulator = new SupplyChainSimulator();

    // Create suppliers
    const supplier1 = new Supplier("supp1", "Raw Materials Inc.", 0.95, 5.0, 2);
    const supplier2 = new Supplier("supp2", "Quality Components Ltd.", 0.98, 7.0, 1);
    simulator.addSupplier(supplier1);
    simulator.addSupplier(supplier2);

    // Create factories
    const factory1 = new Factory("fact1", "Main Production Plant", 100, 15.0, 0.96, 10);
    const factory2 = new Factory("fact2", "Secondary Facility", 50, 16.0, 0.92, 7);
    simulator.addFactory(factory1);
    simulator.addFactory(factory2);

    // Create warehouses
    const warehouse1 = new Warehouse("wh1", "Central Distribution", 2000, 0.8, "ambient", "high");
    const warehouse2 = new Warehouse("wh2", "Regional Storage", 1500, 0.6, "refrigerated", "medium");
    simulator.addWarehouse(warehouse1);
    simulator.addWarehouse(warehouse2);

    // Create shops
    const shop1 = new Shop("shop1", "Downtown Retail", 80, 45.0);
    const shop2 = new Shop("shop2", "Mall Outlet", 120, 42.0);
    const shop3 = new Shop("shop3", "Airport Store", 60, 50.0);
    simulator.addShop(shop1);
    simulator.addShop(shop2);
    simulator.addShop(shop3);

    // Configure some shops with marketing
    shop1.setMarketingBudget(50);
    shop3.setMarketingBudget(100);

    // Enable random events for more realistic simulation
    simulator.setRandomEvents(true);

    console.log("📊 Initial Setup Complete!");
    console.log(`- ${simulator.factories.size} Factories`);
    console.log(`- ${simulator.warehouses.size} Warehouses`);
    console.log(`- ${simulator.shops.size} Shops`);
    console.log(`- ${simulator.suppliers.size} Suppliers\n`);

    // Run simulation for 30 days
    console.log("⏳ Running 30-day simulation...\n");
    const simulationReport = simulator.simulateMultipleDays(30);
    console.log(simulationReport);

    // Show comprehensive statistics
    const stats = simulator.getComprehensiveStats();
    console.log(stats);

    // Show overall summary
    const summary = simulator.getOverallSummary();
    console.log(summary);
  }
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

// Run the demo simulation
SupplyChainSimulator.runDemo();

// Export classes for external use
export {
  Product,
  ProductionStats,
  TransferStats,
  SalesStats,
  MaintenanceEvent,
  MarketEvent,
  Factory,
  Warehouse,
  Shop,
  Supplier,
  SupplyChainSimulator
};