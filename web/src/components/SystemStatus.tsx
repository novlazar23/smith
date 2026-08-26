interface SystemStatusProps {
  health: any
}

export default function SystemStatus({ health }: SystemStatusProps) {
  const services = [
    { name: 'API', status: health?.status === 'ok' ? 'healthy' : 'unknown' },
    { name: 'PostgreSQL', status: 'healthy' },
    { name: 'InfluxDB', status: 'healthy' },
    { name: 'Redis', status: 'healthy' },
  ]

  return (
    <div className="space-y-6">
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">Services</h3>
        <div className="space-y-3">
          {services.map(service => (
            <div key={service.name} className="flex items-center justify-between py-2 border-b border-harness-border last:border-0">
              <span>{service.name}</span>
              <span className={`badge ${
                service.status === 'healthy' ? 'badge-success' : 'badge-warning'
              }`}>
                {service.status}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">Configuration</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Live Execution:</span>
            <span className="ml-2 text-harness-danger">DISABLED</span>
          </div>
          <div>
            <span className="text-gray-400">Shadow Trading:</span>
            <span className="ml-2 text-harness-success">ENABLED</span>
          </div>
          <div>
            <span className="text-gray-400">Kill Switch:</span>
            <span className="ml-2 text-harness-success">ARMED</span>
          </div>
          <div>
            <span className="text-gray-400">Max Leverage:</span>
            <span className="ml-2">1.0x</span>
          </div>
        </div>
      </div>
    </div>
  )
}
