echo "Running network_setup..."
first_line_res=$(ip route show | head -n 1)
sudo ip route change ${first_line_res} rto_min 5ms
sudo ethtool -K ens9 tx-nocache-copy on
echo "network_setup complete!"