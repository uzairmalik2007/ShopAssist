// components/Chat/LaptopCard.tsx
import React from 'react';
import { Laptop } from '../../types';
import { Cpu, Database, Monitor, HardDrive } from 'lucide-react';

interface Props {
  laptop: Laptop;
}

const LaptopCard: React.FC<Props> = ({ laptop }) => {
  return (
    <div className="bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow p-4 border border-gray-200">
      <div className="flex flex-col h-full">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">
          {laptop.brand} {laptop.model_name}
        </h3>

        <div className="space-y-2 flex-grow">
          <div className="flex items-center text-gray-600">
            <Cpu className="h-4 w-4 mr-2" />
            <span className="text-sm">{laptop.core}</span>
          </div>

          <div className="flex items-center text-gray-600">
            <Database className="h-4 w-4 mr-2" />
            <span className="text-sm">{laptop.ram_size} GB RAM</span>
          </div>

          <div className="flex items-center text-gray-600">
            <Monitor className="h-4 w-4 mr-2" />
            <span className="text-sm">{laptop.display_size} inches</span>
          </div>

          <div className="flex items-center text-gray-600">
            <HardDrive className="h-4 w-4 mr-2" />
            <span className="text-sm">{laptop.storage_type}</span>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex justify-between items-center">
            <span className="text-2xl font-bold text-gray-900">
              ₹{laptop.price_inr.toLocaleString()}
            </span>
            <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
              Details
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LaptopCard;