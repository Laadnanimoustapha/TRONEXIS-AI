// Enhanced Supply Chain Simulation System - C++ Version
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>       // For std::shared_ptr, std::make_shared
#include <cmath>        // For std::floor, std::min, std::max
#include <random>       // For std::mt19937, std::random_device, distributions
#include <iomanip>      // For std::setprecision, std::fixed
#include <sstream>      // For std::stringstream
#include <optional>     // For std::optional
#include <algorithm>    // For std::min, std::max

// ============================================================================
// ENUMS, INTERFACES, AND TYPES (as Structs and Enums)
// ============================================================================

struct Product {
    std::string id;
    std::string name;
    double productionCost;
    double sellingPrice;
    double weight; // kg per unit
};

struct ProductionStats {
    int produced;
    double cost;
    int defective;
};

struct TransferStats {
    int sent;
    int received;
    double cost;
};

struct SalesStats {
    int sold;
    double revenue;
    int stockouts;
};

struct MaintenanceEvent {
    int day;
    std::string entityId;
    std::string entityName;
    double cost;
    int downtime;
};

// Enum for Market Event types
enum class MarketEventType {
    DEMAND_SURGE,
    DEMAND_DROP,
    PRICE_INCREASE,
    PRICE_DECREASE,
    SUPPLY_DISRUPTION
};

// Helper function to convert MarketEventType to string
std::string marketEventTypeToString(MarketEventType type) {
    switch (type) {
        case MarketEventType::DEMAND_SURGE: return "demand_surge";
        case MarketEventType::DEMAND_DROP: return "demand_drop";
        case MarketEventType::PRICE_INCREASE: return "price_increase";
        case MarketEventType::PRICE_DECREASE: return "price_decrease";
        case MarketEventType::SUPPLY_DISRUPTION: return "supply_disruption";
        default: return "unknown";
    }
}

struct MarketEvent {
    int day;
    MarketEventType type;
    std::string description;
    std::string affectedEntity;
    double impact;
};

// Enums for Warehouse properties
enum class WarehouseTemperature {
    AMBIENT,
    REFRIGERATED,
    FROZEN
};

std::string warehouseTempToString(WarehouseTemperature temp) {
    switch (temp) {
        case WarehouseTemperature::AMBIENT: return "ambient";
        case WarehouseTemperature::REFRIGERATED: return "refrigerated";
        case WarehouseTemperature::FROZEN: return "frozen";
        default: return "unknown";
    }
}

enum class WarehouseSecurity {
    BASIC,
    MEDIUM,
    HIGH
};

std::string warehouseSecToString(WarehouseSecurity sec) {
    switch (sec) {
        case WarehouseSecurity::BASIC: return "basic";
        case WarehouseSecurity::MEDIUM: return "medium";
        case WarehouseSecurity::HIGH: return "high";
        default: return "unknown";
    }
}

// Stats structs (for getStats() methods)
struct FactoryStats {
    std::string id;
    std::string name;
    int currentStock;
    int productionRate;
    int totalProduced;
    double totalProductionCost;
    int totalDefective;
    double qualityRate;
    double totalMaintenanceCost;
    bool isOperational;
    int downtime;
};

struct WarehouseStats {
    std::string id;
    std::string name;
    int currentStock;
    int capacity;
    double utilization;
    int totalReceived;
    int totalSent;
    double totalStorageCost;
    int totalSpoiled;
    WarehouseTemperature temperature;
    WarehouseSecurity securityLevel;
};

struct ShopStats {
    std::string id;
    std::string name;
    int currentStock;
    int demand;
    double sellingPrice;
    int totalSold;
    double totalRevenue;
    int stockoutDays;
    double customerSatisfaction;
    double totalMarketingSpent;
    int totalReturns;
    double returnRate;
};

struct SupplierSupplyResult {
    int supplied;
    double cost;
    bool success;
};

struct SupplierStats {
    std::string id;
    std::string name;
    double reliability;
    double rawMaterialCost;
    int leadTime;
    int totalSupplied;
    double totalCost;
    int deliveryFailures;
};


// ============================================================================
// FACTORY CLASS (Enhanced)
// ============================================================================

class Factory {
    std::string id;
    std::string name;
    int productionRate;
    int currentStock;
    double productionCostPerUnit;
    double qualityRate; // 0.0 to 1.0 (percentage of non-defective products)
    int maintenanceCycle; // days between maintenance
    int totalProduced = 0;
    double totalProductionCost = 0;
    int totalDefective = 0;
    int daysSinceMaintenanceCheck = 0;
    double totalMaintenanceCost = 0;
    bool isOperational = true;
    int downtime = 0;

public:
    Factory(
        std::string id,
        std::string name,
        int productionRate,
        double productionCostPerUnit,
        double qualityRate = 0.95,
        int maintenanceCycle = 7
    ) : id(id),
        name(name),
        productionRate(productionRate),
        currentStock(0),
        productionCostPerUnit(productionCostPerUnit),
        qualityRate(qualityRate),
        maintenanceCycle(maintenanceCycle) {}

    const std::string& getId() const { return id; }
    const std::string& getName() const { return name; }
    int getCurrentStock() const { return currentStock; }
    int getDemand() const { return productionRate; } // Used for stats

    ProductionStats produce() {
        if (!isOperational) {
            downtime--;
            if (downtime <= 0) {
                isOperational = true;
            }
            return { 0, 0.0, 0 };
        }

        const int produced = productionRate;
        const int goodUnits = static_cast<int>(std::floor(produced * qualityRate));
        const int defective = produced - goodUnits;
        const double cost = produced * productionCostPerUnit;

        currentStock += goodUnits;
        totalProduced += goodUnits;
        totalDefective += defective;
        totalProductionCost += cost;
        daysSinceMaintenanceCheck++;

        return { goodUnits, cost, defective };
    }

    std::optional<MaintenanceEvent> performMaintenance() {
        if (daysSinceMaintenanceCheck >= maintenanceCycle) {
            const double maintenanceCost = productionRate * 2.0;
            totalMaintenanceCost += maintenanceCost;
            daysSinceMaintenanceCheck = 0;
            qualityRate = std::min(0.98, qualityRate + 0.02); // Improve quality

            return MaintenanceEvent{
                0, // Will be set by simulator
                id,
                name,
                maintenanceCost,
                0
            };
        }
        return std::nullopt;
    }

    void setDowntime(int days) {
        isOperational = false;
        downtime = days;
    }

    int transferOut(int amount) {
        const int transferred = std::min(amount, currentStock);
        currentStock -= transferred;
        return transferred;
    }

    void setProductionRate(int rate) {
        productionRate = std::max(0, rate);
    }

    FactoryStats getStats() const {
        return {
            id,
            name,
            currentStock,
            productionRate,
            totalProduced,
            totalProductionCost,
            totalDefective,
            qualityRate,
            totalMaintenanceCost,
            isOperational,
            downtime
        };
    }
};

// ============================================================================
// WAREHOUSE CLASS (Enhanced)
// ============================================================================

class Warehouse {
    std::string id;
    std::string name;
    int capacity;
    int currentStock;
    double storageCostPerUnit;
    WarehouseTemperature temperature;
    WarehouseSecurity securityLevel;
    int totalReceived = 0;
    int totalSent = 0;
    double totalStorageCost = 0;
    double spoilageRate = 0.001; // 0.1% daily spoilage
    int totalSpoiled = 0;
    double insuranceCost = 10.0; // daily insurance cost

public:
    Warehouse(
        std::string id,
        std::string name,
        int capacity,
        double storageCostPerUnit,
        WarehouseTemperature temperature = WarehouseTemperature::AMBIENT,
        WarehouseSecurity securityLevel = WarehouseSecurity::MEDIUM
    ) : id(id),
        name(name),
        capacity(capacity),
        currentStock(0),
        storageCostPerUnit(storageCostPerUnit),
        temperature(temperature),
        securityLevel(securityLevel)
    {
        // Adjust costs based on type
        if (temperature == WarehouseTemperature::REFRIGERATED) {
            this->storageCostPerUnit *= 1.5;
            this->insuranceCost *= 1.3;
        } else if (temperature == WarehouseTemperature::FROZEN) {
            this->storageCostPerUnit *= 2.0;
            this->insuranceCost *= 1.5;
        }

        if (securityLevel == WarehouseSecurity::HIGH) {
            this->insuranceCost *= 1.5;
        }
    }

    const std::string& getId() const { return id; }
    const std::string& getName() const { return name; }
    int getCurrentStock() const { return currentStock; }
    int getCapacity() const { return capacity; }

    int receiveGoods(int amount) {
        const int availableSpace = capacity - currentStock;
        const int received = std::min(amount, availableSpace);

        currentStock += received;
        totalReceived += received;

        return received;
    }

    int transferOut(int amount) {
        const int transferred = std::min(amount, currentStock);
        currentStock -= transferred;
        totalSent += transferred;
        return transferred;
    }

    double calculateStorageCost() {
        const double storageCost = currentStock * storageCostPerUnit;
        const double totalCost = storageCost + insuranceCost;
        totalStorageCost += totalCost;

        // Calculate spoilage
        const int spoiled = static_cast<int>(std::floor(currentStock * spoilageRate));
        currentStock -= spoiled;
        totalSpoiled += spoiled;

        return totalCost;
    }

    void setCapacity(int capacity) {
        this->capacity = std::max(0, capacity);
    }

    WarehouseStats getStats() const {
        return {
            id,
            name,
            currentStock,
            capacity,
            (capacity > 0 ? (static_cast<double>(currentStock) / capacity) : 0.0),
            totalReceived,
            totalSent,
            totalStorageCost,
            totalSpoiled,
            temperature,
            securityLevel
        };
    }
};

// ============================================================================
// SHOP CLASS (Enhanced)
// ============================================================================

class Shop {
    std::string id;
    std::string name;
    int currentStock;
    int demand;
    double sellingPrice;
    double marketingBudget = 0.0;
    double customerSatisfaction = 100.0; // 0-100 scale
    int totalSold = 0;
    double totalRevenue = 0;
    int stockoutDays = 0;
    double totalMarketingSpent = 0;
    double returnRate = 0.02; // 2% return rate
    int totalReturns = 0;

public:
    Shop(
        std::string id,
        std::string name,
        int demand,
        double sellingPrice
    ) : id(id),
        name(name),
        currentStock(0),
        demand(demand),
        sellingPrice(sellingPrice) {}

    const std::string& getId() const { return id; }
    const std::string& getName() const { return name; }
    int getCurrentStock() const { return currentStock; }
    int getDemand() const { return demand; }
    double getMarketingBudget() const { return marketingBudget; }

    int receiveGoods(int amount) {
        currentStock += amount;
        return amount;
    }

    SalesStats sell() {
        const int sellAmount = std::min(demand, currentStock);
        const int stockout = demand - sellAmount;

        // Apply marketing effect (increases demand fulfillment)
        const int marketingBoost = static_cast<int>(std::floor(marketingBudget / 10.0));
        const int boostedSellAmount = std::min(sellAmount + marketingBoost, currentStock);

        // Calculate returns
        const int returns = static_cast<int>(std::floor(boostedSellAmount * returnRate));
        const int netSold = boostedSellAmount - returns;

        currentStock -= boostedSellAmount;
        currentStock += returns; // Returns go back to stock
        totalSold += netSold;
        totalReturns += returns;

        const double revenue = netSold * sellingPrice;
        totalRevenue += revenue;

        // Update customer satisfaction
        if (stockout > 0) {
            stockoutDays += 1;
            customerSatisfaction = std::max(0.0, customerSatisfaction - 5.0);
        } else {
            customerSatisfaction = std::min(100.0, customerSatisfaction + 2.0);
        }

        // Spend marketing budget
        totalMarketingSpent += marketingBudget;

        return {
            netSold,
            revenue,
            stockout
        };
    }

    void setDemand(int demand) {
        this->demand = std::max(0, demand);
    }

    void setSellingPrice(double price) {
        sellingPrice = std::max(0.0, price);
    }

    void setMarketingBudget(double budget) {
        marketingBudget = std::max(0.0, budget);
    }

    ShopStats getStats() const {
        return {
            id,
            name,
            currentStock,
            demand,
            sellingPrice,
            totalSold,
            totalRevenue,
            stockoutDays,
            customerSatisfaction,
            totalMarketingSpent,
            totalReturns,
            returnRate
        };
    }
};

// ============================================================================
// SUPPLIER CLASS (NEW)
// ============================================================================

class Supplier {
    std::string id;
    std::string name;
    double reliability; // 0.0 to 1.0
    double rawMaterialCost;
    int leadTime; // days
    int totalSupplied = 0;
    double totalCost = 0;
    int deliveryFailures = 0;

public:
    Supplier(
        std::string id,
        std::string name,
        double reliability,
        double rawMaterialCost,
        int leadTime
    ) : id(id),
        name(name),
        reliability(reliability),
        rawMaterialCost(rawMaterialCost),
        leadTime(leadTime) {}

    const std::string& getId() const { return id; }
    const std::string& getName() const { return name; }

    SupplierSupplyResult supply(int amount, std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const bool success = dist(rng) < reliability;

        if (!success) {
            deliveryFailures++;
            return { 0, 0.0, false };
        }

        const double cost = amount * rawMaterialCost;
        totalSupplied += amount;
        totalCost += cost;

        return { amount, cost, true };
    }

    SupplierStats getStats() const {
        return {
            id,
            name,
            reliability,
            rawMaterialCost,
            leadTime,
            totalSupplied,
            totalCost,
            deliveryFailures
        };
    }
};

// ============================================================================
// SUPPLY CHAIN SIMULATOR (Enhanced)
// ============================================================================

class SupplyChainSimulator {
    std::unordered_map<std::string, std::shared_ptr<Factory>> factories;
    std::unordered_map<std::string, std::shared_ptr<Warehouse>> warehouses;
    std::unordered_map<std::string, std::shared_ptr<Shop>> shops;
    std::unordered_map<std::string, std::shared_ptr<Supplier>> suppliers;
    double transportCostPerUnit = 0.5;
    int currentDay = 0;
    std::vector<std::string> dailyReports;
    std::vector<MaintenanceEvent> maintenanceEvents;
    std::vector<MarketEvent> marketEvents;
    bool enableRandomEvents = true;
    double totalTransportCost = 0;

    // Random number generator
    std::mt19937 rng;

public:
    SupplyChainSimulator() : rng(std::random_device{}()) {}

    void addFactory(std::shared_ptr<Factory> factory) {
        factories[factory->getId()] = factory;
    }

    void addWarehouse(std::shared_ptr<Warehouse> warehouse) {
        warehouses[warehouse->getId()] = warehouse;
    }

    void addShop(std::shared_ptr<Shop> shop) {
        shops[shop->getId()] = shop;
    }

    void addSupplier(std::shared_ptr<Supplier> supplier) {
        suppliers[supplier->getId()] = supplier;
    }

    void setTransportCost(double cost) {
        transportCostPerUnit = std::max(0.0, cost);
    }

    void setRandomEvents(bool enabled) {
        enableRandomEvents = enabled;
    }

private:
    std::optional<MarketEvent> generateRandomEvent() {
        std::uniform_real_distribution<double> dist_0_1(0.0, 1.0);
        if (!enableRandomEvents || dist_0_1(rng) > 0.15) return std::nullopt;

        std::vector<MarketEventType> eventTypes = {
            MarketEventType::DEMAND_SURGE,
            MarketEventType::DEMAND_DROP,
            MarketEventType::PRICE_INCREASE,
            MarketEventType::PRICE_DECREASE,
            MarketEventType::SUPPLY_DISRUPTION
        };
        std::uniform_int_distribution<int> typeDist(0, eventTypes.size() - 1);
        MarketEventType type = eventTypes[typeDist(rng)];

        std::vector<std::shared_ptr<Shop>> shopVec;
        for (const auto& pair : shops) shopVec.push_back(pair.second);

        std::vector<std::shared_ptr<Factory>> factoryVec;
        for (const auto& pair : factories) factoryVec.push_back(pair.second);

        std::optional<MarketEvent> event = std::nullopt;

        switch (type) {
            case MarketEventType::DEMAND_SURGE:
                if (!shopVec.empty()) {
                    std::uniform_int_distribution<int> shopDist(0, shopVec.size() - 1);
                    auto shop = shopVec[shopDist(rng)];
                    int increase = static_cast<int>(std::floor(shop->getDemand() * 0.3));
                    shop->setDemand(shop->getDemand() + increase);
                    event = MarketEvent{
                        currentDay,
                        type,
                        "Market surge at " + shop->getName() + "! Demand increased by " + std::to_string(increase) + " units.",
                        shop->getId(),
                        static_cast<double>(increase)
                    };
                }
                break;
            case MarketEventType::DEMAND_DROP:
                if (!shopVec.empty()) {
                    std::uniform_int_distribution<int> shopDist(0, shopVec.size() - 1);
                    auto shop = shopVec[shopDist(rng)];
                    int decrease = static_cast<int>(std::floor(shop->getDemand() * 0.2));
                    shop->setDemand(std::max(0, shop->getDemand() - decrease));
                    event = MarketEvent{
                        currentDay,
                        type,
                        "Economic downturn at " + shop->getName() + ". Demand decreased by " + std::to_string(decrease) + " units.",
                        shop->getId(),
                        static_cast<double>(-decrease)
                    };
                }
                break;
            case MarketEventType::SUPPLY_DISRUPTION:
                if (!factoryVec.empty()) {
                    std::uniform_int_distribution<int> factoryDist(0, factoryVec.size() - 1);
                    auto factory = factoryVec[factoryDist(rng)];
                    factory->setDowntime(2);
                    event = MarketEvent{
                        currentDay,
                        type,
                        "Equipment failure at " + factory->getName() + "! Down for 2 days.",
                        factory->getId(),
                        -2.0
                    };
                }
                break;
            // PRICE_INCREASE and PRICE_DECREASE are not implemented in the TS logic,
            // but the types are there. We'll skip them as per the original code.
            default:
                break;
        }

        if (event) {
            marketEvents.push_back(*event);
        }

        return event;
    }

public:
    std::string simulateDay() {
        currentDay++;
        std::stringstream ss;
        std::string separator(80, '=');

        ss << "\n" << separator << "\n";
        ss << "DAY " << currentDay << " SIMULATION\n";
        ss << separator << "\n\n";

        // Check for random events
        auto randomEvent = generateRandomEvent();
        if (randomEvent) {
            ss << "🎲 RANDOM EVENT: " << randomEvent->description << "\n\n";
        }

        double totalProductionCost = 0;
        double totalStorageCost = 0;
        double totalTransportCost = 0;
        double totalRevenue = 0;
        double totalMaintenanceCost = 0;
        double totalMarketingCost = 0;

        // Maintenance Phase
        ss << "--- MAINTENANCE PHASE ---\n";
        bool maintenancePerformed = false;
        for (auto& pair : factories) {
            auto& factory = pair.second;
            if (auto maintenanceOpt = factory->performMaintenance()) {
                MaintenanceEvent maintenance = *maintenanceOpt;
                maintenance.day = currentDay;
                maintenanceEvents.push_back(maintenance);
                totalMaintenanceCost += maintenance.cost;
                ss << "🔧 " << factory->getName() << ": Scheduled maintenance completed (Cost: $"
                   << std::fixed << std::setprecision(2) << maintenance.cost << ")\n";
                maintenancePerformed = true;
            }
        }
        if (!maintenancePerformed) {
            ss << "No maintenance required today.\n";
        }

        // Production Phase
        ss << "\n--- PRODUCTION PHASE ---\n";
        for (auto& pair : factories) {
            auto& factory = pair.second;
            ProductionStats stats = factory->produce();
            totalProductionCost += stats.cost;
            if (stats.produced == 0 && stats.cost == 0) {
                ss << "⚠️  " << factory->getName() << ": NOT OPERATIONAL (Downtime remaining)\n";
            } else {
                ss << factory->getName() << ": Produced " << stats.produced << " units";
                if (stats.defective > 0) {
                    ss << " (" << stats.defective << " defective)";
                }
                ss << " (Cost: $" << std::fixed << std::setprecision(2) << stats.cost << ")\n";
                ss << "  Current Stock: " << factory->getCurrentStock() << " units\n";
            }
        }

        // Logistics Phase (Factory → Warehouse)
        ss << "\n--- LOGISTICS PHASE (Factory → Warehouse) ---\n";
        std::vector<std::shared_ptr<Warehouse>> warehouseVec;
        for (const auto& pair : warehouses) warehouseVec.push_back(pair.second);

        for (auto& pair : factories) {
            auto& factory = pair.second;
            int factoryStock = factory->getCurrentStock();
            if (factoryStock > 0 && !warehouseVec.empty()) {
                int amountPerWarehouse = static_cast<int>(std::floor(static_cast<double>(factoryStock) / warehouseVec.size()));

                for (auto& warehouse : warehouseVec) {
                    const int transferred = factory->transferOut(amountPerWarehouse);
                    if (transferred == 0) continue;
                    
                    const int received = warehouse->receiveGoods(transferred);
                    const double transportCost = transferred * transportCostPerUnit;
                    totalTransportCost += transportCost;
                    this->totalTransportCost += transportCost;

                    ss << factory->getName() << " → " << warehouse->getName() << ": ";
                    if (received < transferred) {
                        ss << "Transferred " << transferred << ", accepted " << received << " (capacity limit).";
                    } else {
                        ss << "Transferred " << transferred << " units.";
                    }
                    ss << " Cost: $" << std::fixed << std::setprecision(2) << transportCost << "\n";
                }
            }
        }

        // Storage Phase
        ss << "\n--- STORAGE PHASE ---\n";
        for (auto& pair : warehouses) {
            auto& warehouse = pair.second;
            const double storageCost = warehouse->calculateStorageCost();
            totalStorageCost += storageCost;
            auto stats = warehouse->getStats();
            ss << warehouse->getName() << ": Storing " << warehouse->getCurrentStock() << "/" << warehouse->getCapacity()
               << " units (" << std::fixed << std::setprecision(1) << (stats.utilization * 100.0) << "% full)\n";
            ss << "  Cost: $" << std::fixed << std::setprecision(2) << storageCost
               << " | Type: " << warehouseTempToString(stats.temperature)
               << " | Security: " << warehouseSecToString(stats.securityLevel) << "\n";
        }

        // Logistics Phase (Warehouse → Shop)
        ss << "\n--- LOGISTICS PHASE (Warehouse → Shop) ---\n";
        for (auto& pair : shops) {
            auto& shop = pair.second;
            const int demandToFulfill = shop->getDemand();
            int remainingDemand = demandToFulfill;

            for (auto& wh_pair : warehouses) {
                auto& warehouse = wh_pair.second;
                if (remainingDemand > 0) {
                    const int transferred = warehouse->transferOut(remainingDemand);
                    shop->receiveGoods(transferred);
                    const double transportCost = transferred * transportCostPerUnit;
                    totalTransportCost += transportCost;
                    this->totalTransportCost += transportCost;
                    remainingDemand -= transferred;

                    if (transferred > 0) {
                        ss << warehouse->getName() << " → " << shop->getName() << ": Transferred " << transferred
                           << " units. Cost: $" << std::fixed << std::setprecision(2) << transportCost << "\n";
                    }
                }
            }

            if (remainingDemand > 0) {
                ss << "⚠️  " << shop->getName() << ": Could not fulfill " << remainingDemand << " units (Stockout)\n";
            }
        }

        // Sales Phase
        ss << "\n--- SALES PHASE ---\n";
        for (auto& pair : shops) {
            auto& shop = pair.second;
            const SalesStats salesStats = shop->sell();
            totalRevenue += salesStats.revenue;
            totalMarketingCost += shop->getMarketingBudget();

            auto shopStats = shop->getStats();
            ss << shop->getName() << ": Sold " << salesStats.sold << " units (Revenue: $"
               << std::fixed << std::setprecision(2) << salesStats.revenue << ")\n";
            ss << "  Customer Satisfaction: " << std::fixed << std::setprecision(1) << shopStats.customerSatisfaction
               << "% | Stock: " << shop->getCurrentStock() << " units\n";
            if (salesStats.stockouts > 0) {
                ss << "  ⚠️  Stockout: " << salesStats.stockouts << " units unmet\n";
            }
        }

        // Financial Summary
        const double totalCost = totalProductionCost + totalStorageCost + totalTransportCost + totalMaintenanceCost + totalMarketingCost;
        const double profit = totalRevenue - totalCost;

        ss << "\n--- FINANCIAL SUMMARY (Day " << currentDay << ") ---\n";
        ss << std::fixed << std::setprecision(2);
        ss << "Production Cost:   $" << std::setw(10) << totalProductionCost << "\n";
        ss << "Storage Cost:      $" << std::setw(10) << totalStorageCost << "\n";
        ss << "Transport Cost:    $" << std::setw(10) << totalTransportCost << "\n";
        ss << "Maintenance Cost:  $" << std::setw(10) << totalMaintenanceCost << "\n";
        ss << "Marketing Cost:    $" << std::setw(10) << totalMarketingCost << "\n";
        ss << "Total Cost:        $" << std::setw(10) << totalCost << "\n";
        ss << "Total Revenue:     $" << std::setw(10) << totalRevenue << "\n";
        ss << "Net Profit/Loss:   $" << std::setw(10) << profit << " " << (profit >= 0 ? "✓" : "✗") << "\n";

        std::string report = ss.str();
        dailyReports.push_back(report);
        return report;
    }

    std::string simulateMultipleDays(int days) {
        std::stringstream fullReport;
        for (int i = 0; i < days; i++) {
            fullReport << simulateDay();
        }
        return fullReport.str();
    }

    std::string getComprehensiveStats() const {
        std::stringstream ss;
        std::string separator(80, '=');

        ss << "\n" << separator << "\n";
        ss << "COMPREHENSIVE SUPPLY CHAIN STATISTICS (After Day " << currentDay << ")\n";
        ss << separator << "\n\n";

        ss << "--- FACTORIES ---\n";
        for (const auto& pair : factories) {
            auto stats = pair.second->getStats();
            ss << stats.name << ":\n";
            ss << "  Production Rate: " << stats.productionRate << " units/day | Quality: "
               << std::fixed << std::setprecision(1) << (stats.qualityRate * 100.0) << "%\n";
            ss << "  Current Stock: " << stats.currentStock << " units | Status: "
               << (stats.isOperational ? "Operational" : ("Down (" + std::to_string(stats.downtime) + " days)")) << "\n";
            ss << "  Total Produced: " << stats.totalProduced << " units | Defective: " << stats.totalDefective << " units\n";
            ss << "  Production Cost: $" << std::fixed << std::setprecision(2) << stats.totalProductionCost
               << " | Maintenance: $" << std::fixed << std::setprecision(2) << stats.totalMaintenanceCost << "\n\n";
        }

        ss << "--- WAREHOUSES ---\n";
        for (const auto& pair : warehouses) {
            auto stats = pair.second->getStats();
            ss << stats.name << " [" << warehouseTempToString(stats.temperature) << ", "
               << warehouseSecToString(stats.securityLevel) << " security]:\n";
            ss << "  Capacity: " << stats.capacity << " units | Utilization: "
               << std::fixed << std::setprecision(1) << (stats.utilization * 100.0) << "%\n";
            ss << "  Current Stock: " << stats.currentStock << " units\n";
            ss << "  Total Received: " << stats.totalReceived << " | Sent: " << stats.totalSent
               << " | Spoiled: " << stats.totalSpoiled << "\n";
            ss << "  Total Storage Cost: $" << std::fixed << std::setprecision(2) << stats.totalStorageCost << "\n\n";
        }

        ss << "--- SHOPS ---\n";
        for (const auto& pair : shops) {
            auto stats = pair.second->getStats();
            ss << stats.name << ":\n";
            ss << "  Daily Demand: " << stats.demand << " units | Price: $"
               << std::fixed << std::setprecision(2) << stats.sellingPrice << "/unit\n";
            ss << "  Current Stock: " << stats.currentStock << " units | Customer Satisfaction: "
               << std::fixed << std::setprecision(1) << stats.customerSatisfaction << "%\n";
            ss << "  Total Sold: " << stats.totalSold << " units | Returns: " << stats.totalReturns
               << " (" << std::fixed << std::setprecision(1) << (stats.returnRate * 100.0) << "%)\n";
            ss << "  Revenue: $" << std::fixed << std::setprecision(2) << stats.totalRevenue
               << " | Marketing Spent: $" << std::fixed << std::setprecision(2) << stats.totalMarketingSpent << "\n";
            ss << "  Stockout Days: " << stats.stockoutDays << "\n\n";
        }

        if (!suppliers.empty()) {
            ss << "--- SUPPLIERS ---\n";
            for (const auto& pair : suppliers) {
                auto stats = pair.second->getStats();
                ss << stats.name << ":\n";
                ss << "  Reliability: " << std::fixed << std::setprecision(1) << (stats.reliability * 100.0)
                   << "% | Lead Time: " << stats.leadTime << " days\n";
                ss << "  Material Cost: $" << std::fixed << std::setprecision(2) << stats.rawMaterialCost << "/unit\n";
                ss << "  Total Supplied: " << stats.totalSupplied << " units | Cost: $"
                   << std::fixed << std::setprecision(2) << stats.totalCost << "\n";
                ss << "  Delivery Failures: " << stats.deliveryFailures << "\n\n";
            }
        }

        if (!marketEvents.empty()) {
            ss << "--- MARKET EVENTS ---\n";
            for (const auto& event : marketEvents) {
                ss << "Day " << event.day << ": " << event.description << "\n";
            }
            ss << "\n";
        }

        return ss.str();
    }

    std::string getOverallSummary() const {
        double totalProduction = 0;
        double totalProductionCost = 0;
        double totalDefective = 0;
        double totalMaintenanceCost = 0;
        for (const auto& pair : factories) {
            auto stats = pair.second->getStats();
            totalProduction += stats.totalProduced;
            totalProductionCost += stats.totalProductionCost;
            totalDefective += stats.totalDefective;
            totalMaintenanceCost += stats.totalMaintenanceCost;
        }

        double totalStorageCost = 0;
        double totalSpoiled = 0;
        for (const auto& pair : warehouses) {
            auto stats = pair.second->getStats();
            totalStorageCost += stats.totalStorageCost;
            totalSpoiled += stats.totalSpoiled;
        }

        double totalSold = 0;
        double totalRevenue = 0;
        double totalStockoutDays = 0;
        double totalMarketingSpent = 0;
        double totalReturns = 0;
        double avgCustomerSatisfaction = 0;
        for (const auto& pair : shops) {
            auto stats = pair.second->getStats();
            totalSold += stats.totalSold;
            totalRevenue += stats.totalRevenue;
            totalStockoutDays += stats.stockoutDays;
            totalMarketingSpent += stats.totalMarketingSpent;
            totalReturns += stats.totalReturns;
            avgCustomerSatisfaction += stats.customerSatisfaction;
        }
        if (!shops.empty()) {
            avgCustomerSatisfaction /= shops.size();
        }

        const double totalCost = totalProductionCost + totalStorageCost + this->totalTransportCost + totalMaintenanceCost + totalMarketingSpent;
        const double netProfit = totalRevenue - totalCost;
        const double profitMargin = totalRevenue > 0 ? (netProfit / totalRevenue) * 100.0 : 0.0;
        
        const double netProduction = totalProduction - totalDefective;
        const double netSellable = netProduction - totalSpoiled;
        const double efficiency = netSellable > 0 ? (totalSold / netSellable) * 100.0 : 0.0;
        const double quality = totalProduction > 0 ? (netProduction / totalProduction) * 100.0 : 0.0;
        const double returnRate = totalSold > 0 ? (static_cast<double>(totalReturns) / totalSold) * 100.0 : 0.0;


        std::stringstream ss;
        std::string separator(80, '=');
        ss << "\n" << separator << "\n";
        ss << "OVERALL SUPPLY CHAIN SUMMARY\n";
        ss << separator << "\n";
        ss << "Simulation Period: " << currentDay << " days\n";
        ss << "Random Events: " << (enableRandomEvents ? "Enabled" : "Disabled")
           << " (" << marketEvents.size() << " events occurred)\n\n";
        ss << "--- OPERATIONS ---\n";
        ss << "Total Produced:      " << totalProduction << " units\n";
        ss << "Total Sold:          " << totalSold << " units\n";
        ss << "Defective Units:     " << totalDefective << " units\n";
        ss << "Spoiled Units:       " << totalSpoiled << " units\n";
        ss << "Returned Units:      " << totalReturns << " units\n";
        ss << "Stockout Days:       " << totalStockoutDays << "\n\n";
        ss << "--- FINANCIAL SUMMARY ---\n";
        ss << std::fixed << std::setprecision(2);
        ss << "Total Revenue:       $" << std::setw(12) << totalRevenue << "\n";
        ss << "Total Costs:         $" << std::setw(12) << totalCost << "\n";
        ss << "  Production:        $" << std::setw(12) << totalProductionCost << "\n";
        ss << "  Storage:           $" << std::setw(12) << totalStorageCost << "\n";
        ss << "  Transport:         $" << std::setw(12) << this->totalTransportCost << "\n";
        ss << "  Maintenance:       $" << std::setw(12) << totalMaintenanceCost << "\n";
        ss << "  Marketing:         $" << std::setw(12) << totalMarketingSpent << "\n";
        ss << "Net Profit:          $" << std::setw(12) << netProfit << "\n";
        ss << "Profit Margin:       " << std::fixed << std::setprecision(1) << profitMargin << "%\n\n";
        ss << "--- PERFORMANCE METRICS ---\n";
        ss << std::fixed << std::setprecision(1);
        ss << "Avg Customer Satisfaction: " << avgCustomerSatisfaction << "%\n";
        ss << "Supply Chain Efficiency:   " << efficiency << "%\n";
        ss << "Quality Rate:              " << quality << "%\n";
        ss << "Return Rate:               " << returnRate << "%\n";

        return ss.str();
    }

    // ============================================================================
    // DEMONSTRATION AND TESTING
    // ============================================================================

    static void runDemo() {
        std::cout << "🚀 SUPPLY CHAIN SIMULATION DEMO\n\n";

        // Create simulator instance
        SupplyChainSimulator simulator;

        // Create suppliers
        auto supplier1 = std::make_shared<Supplier>("supp1", "Raw Materials Inc.", 0.95, 5.0, 2);
        auto supplier2 = std::make_shared<Supplier>("supp2", "Quality Components Ltd.", 0.98, 7.0, 1);
        simulator.addSupplier(supplier1);
        simulator.addSupplier(supplier2);

        // Create factories
        auto factory1 = std::make_shared<Factory>("fact1", "Main Production Plant", 100, 15.0, 0.96, 10);
        auto factory2 = std::make_shared<Factory>("fact2", "Secondary Facility", 50, 16.0, 0.92, 7);
        simulator.addFactory(factory1);
        simulator.addFactory(factory2);

        // Create warehouses
        auto warehouse1 = std::make_shared<Warehouse>("wh1", "Central Distribution", 2000, 0.8, WarehouseTemperature::AMBIENT, WarehouseSecurity::HIGH);
        auto warehouse2 = std::make_shared<Warehouse>("wh2", "Regional Storage", 1500, 0.6, WarehouseTemperature::REFRIGERATED, WarehouseSecurity::MEDIUM);
        simulator.addWarehouse(warehouse1);
        simulator.addWarehouse(warehouse2);

        // Create shops
        auto shop1 = std::make_shared<Shop>("shop1", "Downtown Retail", 80, 45.0);
        auto shop2 = std::make_shared<Shop>("shop2", "Mall Outlet", 120, 42.0);
        auto shop3 = std::make_shared<Shop>("shop3", "Airport Store", 60, 50.0);
        simulator.addShop(shop1);
        simulator.addShop(shop2);
        simulator.addShop(shop3);

        // Configure some shops with marketing
        shop1->setMarketingBudget(50);
        shop3->setMarketingBudget(100);

        // Enable random events for more realistic simulation
        simulator.setRandomEvents(true);

        std::cout << "📊 Initial Setup Complete!\n";
        std::cout << "- " << simulator.factories.size() << " Factories\n";
        std::cout << "- " << simulator.warehouses.size() << " Warehouses\n";
        std::cout << "- " << simulator.shops.size() << " Shops\n";
        std::cout << "- " << simulator.suppliers.size() << " Suppliers\n\n";

        // Run simulation for 30 days
        std::cout << "⏳ Running 30-day simulation...\n";
        std::string simulationReport = simulator.simulateMultipleDays(30);
        std::cout << simulationReport;

        // Show comprehensive statistics
        std::string stats = simulator.getComprehensiveStats();
        std::cout << stats;

        // Show overall summary
        std::string summary = simulator.getOverallSummary();
        std::cout << summary;
    }
};

// ============================================================================
// MAIN EXECUTION
// ============================================================================

int main() {
    // Run the demo simulation
    SupplyChainSimulator::runDemo();
    return 0;
}